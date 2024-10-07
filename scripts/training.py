# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:10 2020

@author: gritti
"""

import tqdm, os, glob
from skimage.io import imread
import numpy as np
import time
from morgana.DatasetTools import io as ioDT
from morgana.MLModel import io as ioML
from morgana.MLModel import train
import matplotlib.pyplot as plt

### define parameters for feature generation for network training
sigmas = [1.0, 5.0, 15.0]
downscaling = 0.25
edge_size = 5
pxl_extract_fraction = 0.25
pxl_extract_bias = 0.4
feature_type = "daisy"  # 'daisy' or 'ilastik'
deep = True  # True: deep learning with Multi Layer Perceptrons; False: Logistic regression

###############################################################################

def train_model(model_folder):
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
    print("##### Generating training set...")
    scaler, train_batches = train.generate_training_set_unet(
        img_train,
        [g.astype(np.uint8) for g in gt_train],
        downscaled_size=(512,512),
        edge_size=edge_size,
        buffer_size=100,
        batch_size=40,
    )
    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        sample_image = (scaler.inverse_transform(sample_image.reshape(-1,1)).reshape(*sample_image.shape)).astype(np.uint8)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        ax[0].imshow(sample_image)
        ax[0].axis("off")
        ax[1].imshow(sample_mask)
        ax[1].axis("off")
        plt.show()
    print("##### Training model...")
    start = time.time()
    classifier = train.train_unet(train_batches, epochs=50)
    print("Models trained in %.3f seconds." % (time.time() - start))
    ioML.save_model(
        model_folder,
        classifier,
        scaler,
        sigmas=sigmas,
        down_shape=downscaling,
        edge_size=edge_size,
        fraction=pxl_extract_fraction,
        feature_mode=feature_type,
        bias=pxl_extract_bias,
        deep=deep,
    )
    print("##### Model saved!")

if __name__ == "__main__":
    model_folder = "/Users/nicholb/Documents/data/organoid_data/fullModel"
    train_model(model_folder)