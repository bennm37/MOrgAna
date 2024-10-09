import os
import tqdm
from skimage.io import imread, imsave
import numpy as np
import scipy.ndimage as ndi
import multiprocessing
from itertools import repeat
from morgana.DatasetTools import io as ioDT
from morgana.MLModel import io as ioML
from morgana.MLModel import predict
import re
import argparse


def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def predict_single_image(f_in, classifier, scaler, params, model="MLP"):
    parent, filename = os.path.split(f_in)
    filename, file_extension = os.path.splitext(filename)
    new_name_classifier = os.path.join(parent, "result_segmentation", filename + "_classifier" + file_extension)
    new_name_watershed = os.path.join(parent, "result_segmentation", filename + "_watershed" + file_extension)

    img = imread(f_in)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    if img.shape[-1] == np.min(img.shape):
        img = np.moveaxis(img, -1, 0)
    img = img[0]

    if not os.path.exists(new_name_classifier):
        if model in ["MLP", "logistic"]:
            pred, prob = predict.predict_image(
                img,
                classifier,
                scaler,
                sigmas=params["sigmas"],
                new_shape_scale=params["down_shape"],
                feature_mode=params["feature_mode"],
                model=model,
            )
        else:
            pred, prob = predict.predict_image_unet(
                img,
                classifier,
                scaler,
                image_size=params["image_size"],
            )
        negative = ndi.binary_fill_holes(pred == 0)
        mask_pred = (pred == 1) * negative
        edge_prob = ((2**16 - 1) * prob[2]).astype(np.uint16)
        mask_pred = mask_pred.astype(np.uint8)
        imsave(new_name_classifier, pred)

    if not os.path.exists(new_name_watershed):
        mask_final = predict.make_watershed(mask_pred, edge_prob)
        imsave(new_name_watershed, mask_final)
    return None


def predict_batch(image_folder, classifier, scaler, params, model="MLP"):
    image_folder = os.path.abspath(image_folder)
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
                    repeat(model),
                ),
            ),
            total=N_img,
        )
    )
    print("All images done!")


def predict_folder(image_folder_nested, classifier, scaler, params, model="MLP"):
    """Opens the inspector for the segmentation results in a folder and all its subfolders."""
    flist = os.listdir(image_folder_nested)
    tifs = [f for f in flist if f.endswith(".tif")]
    tifs = sorted(tifs, key=natural_key)
    if image_folder_nested.split("/")[-1] == "result_segmentation":
        return
    if len(tifs) > 0:
        if "result_segmentation" not in flist:
            os.mkdir(image_folder_nested + "/result_segmentation")
        print("Classifying folder: " + image_folder_nested)
        # predict_batch(image_folder_nested, classifier, scaler, params, model=model)
        [predict_single_image(os.path.join(image_folder_nested, f), classifier, scaler, params, model=model) for f in tifs]
    for folder in flist:
        folder_path = os.path.join(image_folder_nested, folder)
        if os.path.isdir(folder_path):
            predict_folder(folder_path, classifier, scaler, params, model=model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_folder", "-m", default=None)
    p.add_argument("--image_folder", "-i", default=None)
    p.add_argument("--recursive", "-r", default=False)
    args = p.parse_args()
    assert args.model_folder is not None, "Model folder not provided."
    assert args.image_folder is not None, "Image folder not provided."
    assert os.path.exists(args.model_folder), "Model folder not found."
    assert os.path.exists(args.image_folder), "Image folder not found."
    model_folder = args.model_folder
    image_folder = args.image_folder
    model = "unet"
    classifier, scaler, params = ioML.load_model(model_folder, model=model)
    if args.recursive:
        predict_folder(image_folder, classifier, scaler, params)
    else:
        predict_batch([image_folder], classifier, scaler, params, model=model)
