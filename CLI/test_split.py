import argparse
from morgana.MLModel.io import split_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", type=str)
    p.add_argument("--fraction", "-f", type=float, default=0.1)
    args = p.parse_args()
    split_test(args.model_folder, fraction=args.fraction)


if __name__ == "__main__":
    split_test("/Users/nicholb/Documents/data/organoid_data/model_test", fraction=0.1)
