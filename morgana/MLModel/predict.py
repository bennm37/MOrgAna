import time
import numpy as np
from skimage import transform, morphology, measure, segmentation
from sklearn.metrics import classification_report

# import scipy.ndimage as ndi
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import clear_border
import tensorflow as tf
from morgana.ImageTools import processfeatures


def create_features(
    _input,
    scaler,
    gt=np.array([]),
    sigmas=[0.1, 0.5, 1, 2.5, 5, 7.5, 10],
    new_shape_scale=-1,
    feature_mode="ilastik",
    check_time=False,
):

    # read in kwargs
    start = time.time()
    if new_shape_scale != -1:
        shape = (
            int(_input.shape[0] * new_shape_scale),
            int(_input.shape[1] * new_shape_scale),
        )
    else:
        shape = _input.shape
    if check_time:
        print(time.time() - start)

    # resize to new shape
    start = time.time()
    _input = transform.resize(_input.astype(float), shape, preserve_range=True)
    if check_time:
        print(time.time() - start)

    # compute all features and normalize images
    start = time.time()
    _input = processfeatures.get_features(_input, sigmas, feature_mode=feature_mode)
    _input = np.transpose(np.reshape(_input, (_input.shape[0], np.prod(shape))))  # flatten the image
    if check_time:
        print(time.time() - start)

    start = time.time()
    _input = scaler.transform(_input)
    if check_time:
        print(time.time() - start)

    return _input, shape


def predict(_input, classifier, shape=None, check_time=False, gt=np.array([]), model="logistic"):
    if shape is None:
        shape = _input.shape

    # use classifier to predict image
    start = time.time()
    if model in ["MLP", "unet"]:
        y_prob = classifier.predict(_input)  # predict probabilities of every pixel for every class
    else:
        y_prob = classifier.predict_proba(_input)
    y_pred = y_prob.argmax(axis=-1).astype(np.uint8)

    if check_time:
        print(time.time() - start)

    if gt:
        gt = transform.resize(gt, shape, order=0, preserve_range=False)
        gt = 1.0 * (gt > np.min(gt))
        gt = np.reshape(gt, np.prod(shape))
        print(classification_report(gt, y_pred))

    return y_pred, y_prob


def reshape(y_pred, y_prob, original_shape, shape, n_classes=3, check_time=False):

    # reshape image back to 2D
    start = time.time()
    y_pred = np.reshape(y_pred, shape)
    y_prob = np.reshape(np.transpose(y_prob), (n_classes, *shape))
    if check_time:
        print(time.time() - start)

    # resize to new shape
    start = time.time()
    y_pred = transform.resize(y_pred, original_shape, order=0, preserve_range=True)
    y_prob = transform.resize(y_prob, (n_classes, *original_shape), order=0, preserve_range=True)
    if check_time:
        print(time.time() - start)

    return y_pred.astype(np.uint8), y_prob


def predict_image(
    _input,
    classifier,
    scaler,
    gt=np.array([]),
    sigmas=[0.1, 0.5, 1, 2.5, 5, 7.5, 10],
    new_shape_scale=-1,
    feature_mode="ilastik",
    check_time=False,
    model="logistic",
):
    original_shape = _input.shape
    n_classes = 3  # len(classifier.classes_)
    _input, shape = create_features(
        _input,
        scaler,
        gt=np.array([]),
        sigmas=sigmas,
        new_shape_scale=new_shape_scale,
        feature_mode=feature_mode,
        check_time=check_time,
    )
    y_pred, y_prob = predict(_input, classifier, gt=gt, check_time=check_time, shape=shape, model=model)
    y_pred, y_prob = reshape(
        y_pred,
        y_prob,
        original_shape,
        shape,
        n_classes=n_classes,
        check_time=check_time,
    )

    return y_pred.astype(np.uint8), y_prob


def predict_image_unet(_input, scaler, model, image_size=(512, 512)):
    resized = tf.image.resize(_input.reshape(*_input.shape, 1), image_size)
    scaled = scaler.transform(resized.reshape(-1, 1)).reshape(*image_size, 1)
    rgb = tf.image.grayscale_to_rgb(tf.constant([scaled], dtype=tf.float32))
    prob = model.predict(rgb)[0]
    prob = transform.resize(prob, _input.shape)
    pred = np.argmax(prob, axis=-1)
    return pred, prob


def make_watershed(mask, edge, new_shape_scale=-1):

    original_shape = mask.shape

    # read in kwargs
    if new_shape_scale != -1:
        shape = (
            int(mask.shape[0] * new_shape_scale),
            int(mask.shape[1] * new_shape_scale),
        )
    else:
        shape = mask.shape

    mask = transform.resize(mask.astype(float), shape, order=0, preserve_range=False)
    edge = transform.resize(edge, shape, order=0, preserve_range=False)
    edge = ((edge - np.min(edge)) / (np.max(edge) - np.min(edge))) ** 2  # make mountains higher

    # label image and compute weighted center of mass
    labeled_foreground = (mask > np.min(mask)).astype(int)
    properties = measure.regionprops(labeled_foreground, mask)
    if not properties:
        weighted_cm = np.array([shape[0] - 1, shape[1] - 1])
    else:
        weighted_cm = properties[0].weighted_centroid
        weighted_cm = np.array(weighted_cm).astype(np.uint16)

    # move marker to local minimum
    loc_m = morphology.local_minima(np.clip(edge, 0, np.percentile(edge, 90)), connectivity=10, indices=True)
    loc_m = np.transpose(np.stack([loc_m[0], loc_m[1]]))
    dist = [np.linalg.norm(weighted_cm - m) for m in loc_m]
    if len(dist) > 0:
        weighted_cm = loc_m[dist.index(np.min(dist))]

    # move corner marker smallest of 0,0 and nx,ny
    corner = np.array([0, 0])
    if edge[-1, -1] < edge[0, 0]:
        corner = np.array([edge.shape[0] - 1, edge.shape[1] - 1])

    # generate seeds
    markers = np.zeros(edge.shape)
    markers[corner[0], corner[1]] = 1
    markers[weighted_cm[0], weighted_cm[1]] = 2

    # perform watershed
    labels = segmentation.watershed(edge, markers.astype(np.uint))
    labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
    labels = transform.resize(labels, original_shape, order=0, preserve_range=False).astype(np.uint8)
    return labels


def make_mask(pred, area_threshold=200, min_size=200):
    mask_pred = remove_small_objects(pred == 1, min_size=min_size)
    mask_pred = remove_small_holes(mask_pred, area_threshold=area_threshold)
    mask_pred = clear_border(mask_pred)
    return mask_pred
