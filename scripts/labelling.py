import tqdm, os, glob, sys
from morgana.GUIs.manualmask import makeManualMask
from morgana.DatasetTools import io

import PyQt5.QtWidgets


def create_GT_mask(model_folder):
    if not os.path.exists(model_folder):
        print("Warning!")
        print(model_folder, ":")
        print("Model folder not created! Skipping this subfolder.")
        return

    trainingset_folder = os.path.join(model_folder, "trainingset")
    if not os.path.exists(trainingset_folder):
        print("Warning!")
        print(model_folder, ":")
        print("Trainingset images not found! Skipping this subfolder.")
        return
    flist_in = io.get_image_list(trainingset_folder, string_filter="_GT", mode_filter="exclude")
    flist_in.sort()
    flist_gt = io.get_image_list(trainingset_folder, string_filter="_GT", mode_filter="include")
    flist_gt.sort()
    if len(flist_in) == 0:
        print(
            "\n\nWarning, no trainingset!",
            'Selected "'
            + model_folder
            + '" but no trainingset *data* detected. Transfer some images in the "trainingset" folder.',
        )
        return
    if len(flist_in) != len(flist_gt):
        print(
            "\n\nWarning, trainingset incomplete!",
            'Selected "'
            + model_folder
            + '" but not all masks have been created.\nPlease provide manually annotated masks.',
        )

        for f in flist_in:
            fn, ext = os.path.splitext(f)
            mask_name = fn + "_GT" + ext

            if not os.path.exists(mask_name):
                if not PyQt5.QtWidgets.QApplication.instance():
                    app = PyQt5.QtWidgets.QApplication(sys.argv)
                else:
                    app = PyQt5.QtWidgets.QApplication.instance()
                m = makeManualMask(f, subfolder="", fn=fn + "_GT" + ext, wsize=(2000, 2000))
                m.show()
                app.exec_()


if __name__ == "__main__":
    # folder containing trainingSet
    model_folder = "/Users/perezg/Documents/data/2024/240924_organo_segment"
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    create_GT_mask(model_folder)
    app.quit()
    print("All binary masks/ground truth images found. Move to the next step.")

