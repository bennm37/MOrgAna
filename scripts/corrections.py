# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:26 2020

@author: gritti
"""

import os, glob, sys
import PyQt5.QtWidgets
from morgana.GUIs import inspection
from morgana.DatasetTools.segmentation import io as ioSeg
from morgana.DatasetTools import io as ioDT
import cProfile


def correct(image_folder):
    """Opens the inspector for the segmentation results in a folder."""
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    image_folder = os.path.abspath(image_folder)
    ("\n-------------" + image_folder + "------------\n")
    flist_in = ioDT.get_image_list(image_folder)
    n_imgs = len(flist_in)
    if os.path.exists(os.path.join(image_folder, "result_segmentation", "segmentation_params.csv")):
        flist_in, chosen_masks, down_shapes, thinnings, smoothings = ioSeg.load_segmentation_params(
            os.path.join(image_folder, "result_segmentation")
        )
        flist_in = [os.path.join(image_folder, i) for i in flist_in]
    else:
        chosen_masks = ["w" for i in range(n_imgs)]
        down_shapes = [0.50 for i in range(n_imgs)]
        thinnings = [10 for i in range(n_imgs)]
        smoothings = [25 for i in range(n_imgs)]

    save_folder = os.path.join(image_folder, "result_segmentation")
    ioSeg.save_segmentation_params(
        save_folder,
        [os.path.split(fin)[-1] for fin in flist_in],
        chosen_masks,
        down_shapes,
        thinnings,
        smoothings,
    )
    w = inspection.inspectionWindow_20max(image_folder, parent=None, start=0, stop=20)
    w.show()
    app.exec()
    app.quit(q)


def correct_folder(image_folder_nested):
    """Opens the inspector for the segmentation results in a folder and all its subfolders."""
    flist = os.listdir(image_folder_nested)
    tifs = [f for f in flist if f.endswith(".tif")]
    if len(tifs) > 0:
        if "result_segmentation" in flist:
            print("Correcting folder: " + image_folder_nested)
            correct(image_folder_nested)
            return
        else:
            print("No segmentation results found for folder: " + image_folder_nested)
            return
    for folder in flist:
        folder_path = os.path.join(image_folder_nested, folder)
        if os.path.isdir(folder_path):
            correct_folder(folder_path)


if __name__ == "__main__":
    correct_folder(
        "/Users/perezg/Documents/data/2024/240924_organo_segment/241002_small/image_1_MMStack_control_DMSO_1-1.ome_restacked/ROI1"
    )
