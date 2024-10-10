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
    model_folder = "/Users/nicholb/Documents/data/organoid_data/model_unet_test"
    image_folder = "/Users/nicholb/Documents/data/organoid_data/model_unet_test/data"
    classifier, scaler, params = ioML.load_model(model_folder)
    params["model"]
