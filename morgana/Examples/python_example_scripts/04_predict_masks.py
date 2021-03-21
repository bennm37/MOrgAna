# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:26:06 2020

@author: gritti
"""

import os, tqdm, glob
from skimage.io import imread, imsave
import numpy as np
import scipy.ndimage as ndi
import multiprocessing
from itertools import repeat
from morgana.DatasetTools import io as ioDT
import morgana.DatasetTools.multiprocessing.istarmap
from morgana.MLModel import io as ioML
from morgana.MLModel import predict

###############################################################################
# select folder containing all image folders to be analysed
# parent_folder = os.path.join('test_data','2020-09-22_conditions')
parent_folder = os.path.join('/','Volumes','trivedi', 'Jia_Le_Lim', 'morgana_example_datasets', 'gastruloids_ipynb', 'condC')

# find out all image subfolders in parent_folder
folder_names = next(os.walk(parent_folder))[1] 

model_folders = glob.glob(os.path.join(parent_folder,'model_*'))
model_folders_name = [os.path.split(model_folder)[-1] for model_folder in model_folders]

# exclude folders in exclude_folder
exclude_folder = []

image_folders = [g for g in folder_names if not g in model_folders_name + exclude_folder]
image_folders = [os.path.join(parent_folder, i) for i in image_folders]

deep = False # True: deep learning with Multi Layer Perceptrons; False: Logistic regression

###############################################################################
    
if __name__ == '__main__':
    
    for i in range(len(image_folders)):
        
        image_folder = image_folders[i]
        if len(model_folders)>1:
            model_folder = model_folders[i]
        else:
            model_folder = model_folders[0]

        print('-------------'+image_folder+'------------')
        print('##### Loading classifier model and parameters...')
        classifier, scaler, params = ioML.load_model( model_folder, deep = deep)
        print('##### Model loaded!')

        #######################################################################
        ### apply classifiers and save images

        result_folder = os.path.join(image_folder, 'result_segmentation')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        flist_in = ioDT.get_image_list(image_folder)
        flist_in.sort()        
        N_img = len(flist_in)

        # multiprocess
        N_cores = np.clip( int(0.8 * multiprocessing.cpu_count()),1,None )

        # try using multiprocessing
        pool = multiprocessing.Pool(N_cores)
        _ = list(   tqdm.tqdm(
                                pool.istarmap(
                                    predict.predict_single_image, 
                                    zip(    flist_in, 
                                            repeat(classifier),
                                            repeat(scaler),
                                            repeat(params) ) ), 
                                    total = N_img ) )

    print('All images done!')