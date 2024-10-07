import os
import pandas as pd
import re


def save_segmentation_params(save_folder, filename, chosen_mask, down_shape, thinning, smoothing):

    # Make it work for Python 2+3 and with Unicode
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    params = pd.DataFrame(
        {
            "filename": filename,
            "chosen_mask": chosen_mask,
            "down_shape": down_shape,
            "thinning": thinning,
            "smoothing": smoothing,
        }
    )
    column_order = [
        "filename",
        "chosen_mask",
        "down_shape",
        "thinning",
        "smoothing",
    ]

    def natural_key(textlist):
        return [[int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)] for text in textlist]

    params = params.sort_values(by="filename", key=natural_key)
    params[column_order].to_csv(os.path.join(save_folder, "segmentation_params.csv"))


def load_segmentation_params(save_folder):

    # with open(os.path.join(save_folder,'segmentation_params.json'), 'r') as f:
    #     params = json.load(f)
    params = pd.read_csv(os.path.join(save_folder, "segmentation_params.csv"))

    def natural_key(textlist):
        return [[int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)] for text in textlist]

    params = params.sort_values(by="filename", key=natural_key)
    filename = params["filename"]
    chosen_mask = params["chosen_mask"]
    down_shape = params["down_shape"]
    for i in range(len(down_shape)):
        if down_shape[i] == 500:
            down_shape[i] = 500.0 / 2160.0
    thinning = params["thinning"]
    smoothing = params["smoothing"]
    return filename, chosen_mask, down_shape, thinning, smoothing
