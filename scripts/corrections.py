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

parent_folder = "/Users/nicholb/Documents/data/organoid_data/240924_model_copy"
image_folders = ["/Users/nicholb/Documents/data/organoid_data/240924_model/model_copy/data"]

if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    for image_folder in image_folders:

    ### compute parent folder as absolute path
        image_folder = os.path.abspath(image_folder)

        ("\n-------------" + image_folder + "------------\n")

        flist_in = ioDT.get_image_list(image_folder)
        n_imgs = len(flist_in)
        if os.path.exists(
            os.path.join(image_folder, "result_segmentation", "segmentation_params.csv")
        ):
            flist_in, chosen_masks, down_shapes, thinnings, smoothings = (
                ioSeg.load_segmentation_params(os.path.join(image_folder, "result_segmentation"))
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
    app.quit()


