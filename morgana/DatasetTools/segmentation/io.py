import os
import pandas as pd
from morgana.DatasetTools.io import natural_key_list


def save_segmentation_params(save_folder, filename, chosen_mask, down_shape, thinning, smoothing):
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
    params = params.sort_values(by="filename", key=natural_key_list)
    params[column_order].to_csv(os.path.join(save_folder, "segmentation_params.csv"))


def load_segmentation_params(save_folder):
    params = pd.read_csv(os.path.join(save_folder, "segmentation_params.csv"))
    params = params.sort_values(by="filename", key=natural_key_list)
    filename = params["filename"]
    chosen_mask = params["chosen_mask"]
    down_shape = params["down_shape"]
    for i in range(len(down_shape)):
        if down_shape[i] == 500:
            down_shape[i] = 500.0 / 2160.0
    thinning = params["thinning"]
    smoothing = params["smoothing"]
    return filename, chosen_mask, down_shape, thinning, smoothing
