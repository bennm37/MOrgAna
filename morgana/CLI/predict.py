import os
from morgana.MLModel import io as ioML
from morgana.MLModel.predict import predict_folder
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", default=None)
    p.add_argument("image_folder", default=None)
    args = p.parse_args()
    assert args.model_folder is not None, "Model folder not provided."
    assert args.image_folder is not None, "Image folder not provided."
    assert os.path.exists(args.model_folder), "Model folder not found."
    assert os.path.exists(args.image_folder), "Image folder not found."
    model_folder = args.model_folder
    image_folder = args.image_folder
    classifier, scaler, params = ioML.load_model(model_folder)
    predict_folder(image_folder, classifier, scaler, params, params["model"])


if __name__ == "__main__":
    # main()
    model_folder = "/camp/lab/vincentj/home/users/nicholb/organoid_data/model_unet"
    image_folder = "/Volumes/lab-vincentj/home/users/nicholb/organoid_data/240930_saving/image_1_MMStack_control_DMSO_1-1.ome_restacked/ROI1"
    classifier, scaler, params = ioML.load_model(model_folder)
    predict_folder(image_folder, classifier, scaler, params, params["model"])
