from morgana.MLModel.io import new_model
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", type=str)
    args = p.parse_args()
    print(args.model_folder)
    new_model(args.model_folder)
