# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:10 2020

@author: gritti
"""

import os
from skimage.io import imread
import numpy as np
import time
from morgana.DatasetTools import io as ioDT
from morgana.MLModel import io as ioML
from morgana.MLModel import train
import argparse
import json


###############################################################################


def train_model(model, model_folder, epochs=50, steps_per_epoch=10, **params):
    print(f"Training model {model} in folder {model_folder} ...")
    default_path = os.path.join(os.path.dirname(__file__), f"../MLModel/default_params/{model}.json")
    with open(default_path, "r") as f:
        default_params = json.load(f)
    params = {key: params.get(key, default_params[key]) for key in default_params.keys()}
    print("-------------" + model_folder + "------------")
    training_folder = os.path.join(model_folder, "trainingset")
    flist_in = ioDT.get_image_list(training_folder, string_filter="_GT", mode_filter="exclude")
    img_train = []
    for f in flist_in:
        img = imread(f)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        img_train.append(img[0])
    flist_gt = ioDT.get_image_list(training_folder, string_filter="_GT", mode_filter="include")
    gt_train = [imread(f) for f in flist_gt]
    gt_train = [g.astype(int) for g in gt_train]

    print("##### Training set:")
    for i, f in enumerate(zip(flist_in, flist_gt)):
        print(
            i + 1,
            "\t",
            os.path.split(f[0])[-1],
            "\t",
            os.path.split(f[1])[-1],
        )
    if model in ["MLP", "logistic"]:
        X, Y, w, scaler = train.generate_training_set(
            img_train,
            [g.astype(np.uint8) for g in gt_train],
            sigmas=params.get("sigmas", [1.0, 5.0, 15.0]),
            down_shape=params.get("downscaling", 0.25),
            edge_size=params.get("edge_size", 5),
            fraction=params.get("pxl_extract_fraction", 0.25),
            feature_mode=params.get("feature_type", "daisy"),
            bias=params.get("pxl_extract_bias", 0.4),
        )
        print("##### Training model...")
        start = time.time()
        classifier = train.train_classifier(X, Y, w, model=model, epochs=epochs, steps_per_epoch=steps_per_epoch)
        print("Models trained in %.3f seconds." % (time.time() - start))
        ioML.save_model(
            model_folder,
            classifier,
            scaler,
            sigmas=params.get("sigmas", [1.0, 5.0, 15.0]),
            down_shape=params.get("downscaling", 0.25),
            edge_size=params.get("edge_size", 5),
            fraction=params.get("pxl_extract_fraction", 0.25),
            feature_mode=params.get("feature_type", "daisy"),
            bias=params.get("pxl_extract_bias", 0.4),
            model=model,
        )
    elif model == "unet":
        print("##### Generating training set...")
        scaler, train_batches = train.generate_training_set_unet(
            img_train,
            [g.astype(np.uint8) for g in gt_train],
            downscaled_size=params.get("downscaled_size", (512, 512)),
            edge_size=params.get("edge_size", 5),
            buffer_size=params.get("buffer_size", 100),
            batch_size=params.get("batch_size", 32),
        )
        print("##### Training model...")
        start = time.time()
        classifier = train.train_unet(train_batches, epochs=epochs, steps_per_epoch=steps_per_epoch)
        print("Models trained in %.3f seconds." % (time.time() - start))
        ioML.save_model(
            model_folder,
            classifier,
            scaler,
            downscaled_size=params.get("downscaled_size", (512, 512)),
            edge_size=params.get("edge_size", 5),
            buffer_size=params.get("buffer_size", 100),
            batch_size=params.get("batch_size", 32),
            model=model,
        )
    else:
        raise ValueError("Model not recognized.")
    print("##### Model saved!")
    return classifier, scaler, params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", type=str, help="Folder containing the training set")
    p.add_argument("--epochs", type=int, help="Number of epochs for training", default=50)
    args = p.parse_args()
    params = ioML.load_params(args.model_folder)
    try:
        model = params["model"]
    except KeyError:
        raise ValueError("Model not specified in the params.json file.")
    train_model(model, args.model_folder, args.epochs)


if __name__ == "__main__":
    main()
