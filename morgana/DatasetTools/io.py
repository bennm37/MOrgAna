import os, glob
import numpy as np
import re

def get_image_list(input_folder, string_filter="", mode_filter="include"):

    flist = glob.glob(os.path.join(input_folder, "*.tif"))
    if string_filter:
        if mode_filter == "include":
            flist = [f for f in flist if string_filter in f]
        elif mode_filter == "exclude":
            flist = [f for f in flist if string_filter not in f]
    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    flist = sorted([f for f in flist if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))], key=natural_key)
    return flist



 