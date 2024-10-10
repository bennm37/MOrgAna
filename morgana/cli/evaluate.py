from morgana.MLModel.evaluate import evaluate
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", type=str)
    args = p.parse_args()
    evaluate(args.model_folder)


if __name__ == "__main__":
    model_folder = "/Users/nicholb/Documents/data/organoid_data/fullModel"
    evaluate(model_folder)
