import os
import glob
import re


def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def get_image_list(input_folder, string_filter="", mode_filter="include"):

    flist = glob.glob(os.path.join(input_folder, "*.tif"))
    if string_filter:
        if mode_filter == "include":
            flist = [f for f in flist if string_filter in f]
        elif mode_filter == "exclude":
            flist = [f for f in flist if string_filter not in f]
    flist = sorted(
        [f for f in flist if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))], key=natural_key
    )
    return flist


def apply_nested(f, parent_folder):
    """Takes a function that can be applied to a folder of tifs and applies it recursively to all subfolders."""
    flist = os.listdir(parent_folder)
    tifs = [f for f in flist if f.endswith(".tif")]
    if len(tifs) > 0:
        if "result_segmentation" in flist:
            print("Correcting folder: " + parent_folder)
            f(parent_folder)
            return
        else:
            print("No segmentation results found for folder: " + parent_folder)
            return
    for folder in flist:
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            apply_nested(f, folder_path)
