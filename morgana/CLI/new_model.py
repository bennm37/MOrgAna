from morgana.MLModel.io import new_model
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", "-m", type=str, default="unet")
    p.add_argument("model_folder", type=str)
    args = p.parse_args()
    print(args.model_folder)
    new_model(args.model_folder, model=args.model_type)


if __name__ == "__main__":
    main()
