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
import re

def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def predict_single_image(f_in, classifier, scaler, params, deep=True):

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
            deep=deep,
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


def predict_batch(image_folders, model_folder, deep=False):
    for image_folder in image_folders:
        image_folder = os.path.abspath(image_folder)
        print("-------------" + image_folder + "------------")
        training_folder = os.path.join(model_folder, "trainingset")
        print("##### Loading classifier model and parameters...")
        classifier, scaler, params = ioML.load_model(model_folder, deep=deep)
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
                        repeat(deep),
                    ),
                ),
                total=N_img,
            )
        )
        print("All images done!")


def predict_folders_(image_folder_nested, model_folder):
    classifier, scaler, params = ioML.load_model(model_folder, deep=True)

    for folder in os.listdir(image_folder_nested):
        folder_path = os.path.join(image_folder_nested, folder)
        
        # Ensure it's a folder
        if os.path.isdir(folder_path):
            # Iterate over each ROI subfolder
            for roi_subfolder in os.listdir(folder_path):
                roi_path = os.path.join(folder_path, roi_subfolder)
                
                # Ensure it's a subfolder
                if os.path.isdir(roi_path):
                    # List all images in the ROI subfolder
                    image_files = sorted([f for f in os.listdir(roi_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))], key=natural_key)
                    if not os.path.exists(roi_path+'/result_segmentation'):
                        os.mkdir(roi_path+'/result_segmentation')

                    # Copy selected images to the destination folder
                    for image in image_files:
                        src_image_path = os.path.join(roi_path, image)
                        
                        print(f"working with {src_image_path}")
                        predict_single_image(src_image_path, classifier, scaler, params)



def predict_folders_batch(image_folder_nested, model_folder, deep=True):
    classifier, scaler, params = ioML.load_model(model_folder, deep=True)
    
    list_images= []

    for folder in os.listdir(image_folder_nested):
        folder_path = os.path.join(image_folder_nested, folder)
        
        # Ensure it's a folder
        if os.path.isdir(folder_path):
            # Iterate over each ROI subfolder
            for roi_subfolder in os.listdir(folder_path):
                roi_path = os.path.join(folder_path, roi_subfolder)
                
                # Ensure it's a subfolder
                if os.path.isdir(roi_path):
                    # List all images in the ROI subfolder
                    image_files = sorted([os.path.join(roi_path,f) for f in os.listdir(roi_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))], key=natural_key)
                    list_images.extend(image_files)
                    if not os.path.exists(roi_path+'/result_segmentation'):
                        os.mkdir(roi_path+'/result_segmentation')
    
    N_img = len(list_images)
    N_cores = np.clip(int(0.8 * multiprocessing.cpu_count()), 1, None)
    pool = multiprocessing.Pool(N_cores)
    _ = list(
        tqdm.tqdm(
            pool.istarmap(
                predict_single_image,
                zip(
                    list_images,
                    repeat(classifier),
                    repeat(scaler),
                    repeat(params),
                    repeat(deep),
                ),
            ),
            total=N_img,
        )
    )
    




if __name__ == "__main__":
    model_folder = "/Users/perezg/Documents/data/2024/240924_organo_segment"
    image_folder_nested = "/Users/perezg/Documents/data/2024/240924_organo_segment/241002_small"
    predict_folders_batch(image_folder_nested, model_folder)
    #classifier, scaler, params = ioML.load_model(model_folder, deep=True)
    #image_folders = [f"{model_folder}/data"]
    #predict_batch(image_folders, model_folder, deep=True) # will crash is deep is incorrect

