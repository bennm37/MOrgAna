import os, tqdm
from skimage.io import imread, imsave
import numpy as np
import scipy.ndimage as ndi
import multiprocessing
from itertools import repeat
from morgana.DatasetTools import io as ioDT
import morgana.DatasetTools.multiprocessing.istarmap
from morgana.MLModel import io as ioML
from morgana.MLModel import predict


def predict_single_image(f_in, classifier, scaler, params):

    parent, filename = os.path.split(f_in)
    filename, file_extension = os.path.splitext(filename)
    new_name_classifier = os.path.join(
        parent, "result_segmentation", filename + "_classifier" + file_extension
    )
    new_name_watershed = os.path.join(
        parent, "result_segmentation", filename + "_watershed" + file_extension
    )

    img = imread(f_in)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    img = img[0]

    if not os.path.exists(new_name_classifier):
        pred, prob = predict.predict_image(
            img,
            classifier,
            scaler,
            sigmas=params["sigmas"],
            new_shape_scale=params["down_shape"],
            feature_mode=params["feature_mode"],
        )
        negative = ndi.binary_fill_holes(pred == 0)
        mask_pred = (pred == 1) * negative
        edge_prob = ((2**16 - 1) * prob[2]).astype(np.uint16)
        mask_pred = mask_pred.astype(np.uint8)
        imsave(new_name_classifier, pred)

    if not os.path.exists(new_name_watershed):
        mask_final = predict.make_watershed(
            mask_pred, edge_prob, new_shape_scale=params["down_shape"]
        )
        imsave(new_name_watershed, mask_final)

    return None


def predict_batch(image_folders, model_folder):
    for image_folder in image_folders:
        image_folder = os.path.abspath(image_folder)
        print("-------------" + image_folder + "------------")
        training_folder = os.path.join(model_folder, "trainingset")
        print("##### Loading classifier model and parameters...")
        classifier, scaler, params = ioML.load_model(model_folder)
        print("##### Model loaded!")
        result_folder = os.path.join(image_folder, "result_segmentation")
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        flist_in = ioDT.get_image_list(image_folder)
        flist_in.sort()
        N_img = len(flist_in)
        N_cores = np.clip(int(0.8 * multiprocessing.cpu_count()), 1, None)
        pool = multiprocessing.Pool(N_cores)
        _ = list(
            tqdm.tqdm(
                pool.istarmap(
                    predict_single_image,
                    zip(
                        flist_in,
                        repeat(classifier),
                        repeat(scaler),
                        repeat(params),
                    ),
                ),
                total=N_img,
            )
        )
        print("All images done!")


if __name__ == "__main__":
    image_folders = ["/Users/nicholb/Documents/data/organoid_data/240924_model/model_copy/data"]
    model_folder = "/Users/nicholb/Documents/data/organoid_data/240924_model/model"
    predict_batch(image_folders, model_folder)
