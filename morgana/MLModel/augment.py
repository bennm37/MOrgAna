import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()  


class Augment(tf.keras.layers.Layer):
    def __init__(
        self,
        seed=None,
        aug={"flip": "horizontal", "rotation": 0.1, "zoom": (0, -0.2), "translation": 0.02},
    ):
        super(Augment, self).__init__()
        # Define all augmentation layers
        self.input_augmentations = self.get_augemntations(seed, aug)
        self.label_augmentations = self.get_augemntations(seed, aug)

    def get_augemntations(self, seed, aug):
        layers = []
        if "translation" in aug:
            layers.append(
                tf.keras.layers.RandomTranslation(
                    height_factor=aug["translation"],
                    width_factor=aug["translation"],
                    seed=seed,
                )
            )
        if "flip" in aug:
            layers.append(tf.keras.layers.RandomFlip(mode=aug["flip"], seed=seed))
        if "rotation" in aug:
            layers.append(tf.keras.layers.RandomRotation(factor=aug["rotation"], seed=seed, fill_mode="reflect"))
        if "zoom" in aug:
            layers.append(tf.keras.layers.RandomZoom(height_factor=aug["zoom"], width_factor=aug["zoom"], seed=seed))
        return layers

    def call(self, inputs, labels):
        # Apply all augmentations to both inputs and labels
        for input_aug, label_aug in zip(self.input_augmentations, self.label_augmentations):
            inputs = input_aug(inputs)
            labels = label_aug(labels)
        return inputs, labels
