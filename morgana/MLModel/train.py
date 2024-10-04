import tqdm, os
import numpy as np
from skimage import transform, morphology
from sklearn import preprocessing, linear_model, neural_network
from morgana.ImageTools import processfeatures
import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior()


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=None):
        super(Augment, self).__init__()
        # Define all augmentation layers
        self.input_augmentations = [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.1, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed),
            tf.keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.1, seed=seed
            ),
        ]
        self.label_augmentations = [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.1, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed),
            tf.keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.1, seed=seed
            ),
        ]

    def call(self, inputs, labels):
        # Apply all augmentations to both inputs and labels
        for input_aug, label_aug in zip(self.input_augmentations, self.label_augmentations):
            inputs = input_aug(inputs)
            labels = label_aug(labels)
        return inputs, labels


def downsize_image(image, down_shape):
    if down_shape != -1:
        return transform.resize(
            image,
            (
                tf.constant(image.shape[0] * down_shape, dtype=tf.int32),
                tf.constant(image.shape[1] * down_shape, dtype=tf.int32),
            ),
            preserve_range=True,
        )
    else:
        return image


def resize_data(image, label, size):
    return tf.image.resize(image, size), tf.image.resize(label, size)


def extract_edges(gt, edge_size):
    Y = 1.0 * (gt > np.min(gt))
    edge = Y - morphology.binary_dilation(Y, morphology.disk(1))
    edge = morphology.binary_dilation(edge, morphology.disk(edge_size))
    Y = 1 * np.logical_or(Y, edge) + edge
    return Y


def generate_training_set(
    _input,
    gt,
    sigmas=[0.1, 0.5, 1, 2.5, 5, 7.5, 10],
    down_shape=-1,
    edge_size=5,
    fraction=0.1,
    bias=-1,
    edge_weight=5,
    feature_mode="ilastik",
):
    """
    Note: _input and gt should have shape (n_images,x,y)

    """
    _input = [downsize_image(i, down_shape) for i in _input]
    gt = [downsize_image(i, down_shape) for i in gt]
    shapes = [i.shape for i in _input]
    n_coords_per_image = [(fraction * np.prod(i.shape)).astype(int) for i in _input]
    n_coords = int(np.sum(n_coords_per_image))
    print("Number of images: %d" % len(_input))
    print(
        "Number of pixels extracted per image (%d%%):" % (100 * fraction),
        n_coords_per_image,
    )
    if feature_mode == "ilastik":
        print("Number of features per image: %d" % (len(sigmas) * 4 + 1))
        X_train = np.zeros((n_coords, len(sigmas) * 4 + 1))
    elif feature_mode == "daisy":
        print(
            "Number of features per image:%d" % ((5 * 8 + 1) * 8 + len(sigmas) * 4 + 1)
        )
        X_train = np.zeros((n_coords, (5 * 8 + 1) * 8 + len(sigmas) * 4 + 1))
    Y_train = np.zeros(n_coords)
    weight_train = np.zeros(n_coords)

    print("Extracting features...")
    start = 0
    for i in tqdm.tqdm(range(len(_input))):
        stop = start + n_coords_per_image[i]
        shape = shapes[i]
        x_in, y_in = _input[i], gt[i]

        # compute all features
        X = processfeatures.get_features(x_in, sigmas, feature_mode=feature_mode)
        Y = extract_edges(y_in, edge_size)
        # flatten the images
        X = np.transpose(
            np.reshape(X, (X.shape[0], np.prod(shape)))
        )  # flatten the image feature
        Y = np.reshape(Y, np.prod(shape))  # flatten the ground truth
        edge = np.reshape(edge, np.prod(shape))  # flatten the edge

        # extract coordinates with the right probability distribution
        if (bias > 0) and (bias <= 1):
            prob = (Y > 0).astype(float)
            Nw = np.sum(prob)
            Nd = np.prod(prob.shape) - Nw
            probW = bias * prob / Nw
            probD = (1 - bias) * (prob == 0) / Nd
            prob = probW + probD
        else:
            prob = np.ones(Y.shape) / np.prod(Y.shape)
        coords = np.random.choice(np.arange(X.shape[0]), n_coords_per_image[i], p=prob)

        # populate training dataset, ground truth and weight
        X_train[start:stop, :] = X[coords, :]
        Y_train[start:stop] = Y[coords]
        weight = edge_weight * edge + 1
        weight_train[start:stop] = weight[coords]
        start = n_coords_per_image[i]

    scaler = preprocessing.RobustScaler(quantile_range=(1.0, 99.0))
    scaler.fit(X_train)  # normalize
    X_train = scaler.transform(X_train)
    # shuffle the training set
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p, :]
    Y_train = Y_train[p]
    return X_train, Y_train, weight_train, scaler


def generate_training_set_unet(
    _input,
    gt,
    downscaled_size=(512, 512),
    edge_size=5,
    buffer_size=32,
    batch_size=32,
):
    scaler = preprocessing.RobustScaler(quantile_range=(1.0, 99.0))
    scaler.fit(np.array(_input).reshape(-1, 1))
    _input = [scaler.transform(img.reshape(-1,1)).reshape(*img.shape,1) for img in _input]
    labels = [extract_edges(g, edge_size).reshape(*g.shape,1) for g in gt]
    dataset = tf.data.Dataset.from_tensor_slices((_input, labels))
    dataset = dataset.map(lambda x, y: resize_data(x, y, downscaled_size))
    dataset = dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
    train_batches = (
        dataset.shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=0))
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return scaler, train_batches


def train_classifier(X, Y, w, deep=False, epochs=50, n_classes=3, hidden=(350, 50)):
    # train the classifier
    if not deep:
        print("Training of Logistic Regression classifier...")
        classifier = linear_model.LogisticRegression(solver="lbfgs", multi_class="auto")
        classifier.fit(X, Y, sample_weight=w)
    else:
        print("Training of MLP classifier...")
        from tensorflow.keras import layers
        from tensorflow import keras

        Y = keras.utils.to_categorical(Y, num_classes=n_classes)
        model_layers = [
            layers.Dense(hidden[i], activation="relu", name="layer%d" % i)
            for i in range(len(hidden))
        ]
        model_layers.append(
            layers.Dense(n_classes, activation="softmax", name="layer%d" % len(hidden))
        )

        classifier = keras.Sequential(model_layers)
        classifier.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        classifier.fit(
            X,
            Y,
            epochs=epochs,
            batch_size=1024,
            verbose=1,
            validation_split=0.1,
            shuffle=True,
        )
    return classifier


def train_unet(dataset, epochs=50, n_output_classes=3, input_shape=(512, 512, 3)):
    from tensorflow.keras import layers
    from tensorflow import keras
    from tensorflow_examples.models.pix2pix import pix2pix

    # Use a pretrained encoder mobilenetv2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False
    )
    # Use the activations of these layers
    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    def unet_model(output_channels: int):
        inputs = tf.keras.layers.Input(shape=input_shape)
        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2, padding="same"
        )  # 64x64 -> 128x128
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    model = unet_model(output_channels=1)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset, epochs=epochs)
