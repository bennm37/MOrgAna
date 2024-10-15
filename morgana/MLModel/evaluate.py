import os
from morgana.MLModel import io as ioML
from morgana.MLModel import predict
from morgana.DatasetTools import io as ioDT
from skimage.io import imread
import pandas as pd
import numpy as np
from datetime import datetime
import lazy_import
tf = lazy_import.lazy_module("tensorflow")

def evaluate(model_folder):
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    assert ioML.folder_labeled(f"{model_folder}/testset")
    classifier, scaler, params = ioML.load_model(model_folder)
    flist_in = ioDT.get_image_list(os.path.join(model_folder, "testset"), string_filter="_GT", mode_filter="exclude")
    flist_GT = ioDT.get_image_list(os.path.join(model_folder, "testset"), string_filter="_GT", mode_filter="include")
    img_in = [imread(f) for f in flist_in]
    img_GT = [imread(f) for f in flist_GT]
    # img_GT = [transform.resize(g, params["downscaled_size"]) for g in img_GT]
    if params["model"] == "unet":
        img_classifier = [predict.predict_image_unet(img, classifier, scaler) for img in img_in]
        img_masks = [predict.make_mask(c[0]) for c in img_classifier]
        img_watershed = [
            predict.make_watershed(predict.make_mask(m), c[1][:, :, 2]) for m, c in zip(img_masks, img_classifier)
        ]
    else:
        pi_keywords = ["down_shape", "feature_mode", "sigmas", "model"]
        pi_params = {k: v for k, v in params.items() if k in pi_keywords}
        img_classifier = [predict.predict_image(img, classifier, scaler, **pi_params) for img in img_in]
        img_classifier = [(c[0], np.moveaxis(c[1], 0, 2)) for c in img_classifier]
        img_GT = [tf.keras.utils.to_categorical(gt, num_classes=3) for gt in img_GT]
        img_masks = [predict.make_mask(c[0]) for c in img_classifier]
        img_watershed = [
            predict.make_watershed(predict.make_mask(m), c[1][:, :, 2]) for m, c in zip(img_masks, img_classifier)
        ]
    losses = [classifier.loss(g, c[1]).numpy() for c, g in zip(img_classifier, img_GT)]
    classifier_accuracies = [tf.keras.metrics.Accuracy()(g, c[0]).numpy() for c, g in zip(img_classifier, img_GT)]
    watershed_accuracies = [tf.keras.metrics.Accuracy()(g, w).numpy() for w, g in zip(img_watershed, img_GT)]
    index = [f.split("/")[-1][:-4] for f in flist_in]
    df = pd.DataFrame(
        np.array([losses, classifier_accuracies, watershed_accuracies]).T,
        columns=["losses", "classifier_accuracies", "watershed_accuracies"],
        index=index,
    )
    results_folder = f"{model_folder}/testset/results"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    date = datetime.now().strftime("%d-%m-%Y%H:%M:%S")
    print(df)
    df.to_csv(f"{results_folder}/metrics.csv")
    return df
