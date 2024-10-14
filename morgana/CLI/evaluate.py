from morgana.MLModel.evaluate import evaluate
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(losses, classifier_accuracies, watershed_accuracies):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    bins = np.linspace(0, 1, 30)
    ax[0].hist(losses, bins=bins)
    ax[0].set_title("Losses")
    ax[1].hist(classifier_accuracies, bins=bins)
    ax[1].set_title("Classifier Accuracies")
    ax[2].hist(watershed_accuracies, bins=bins)
    ax[2].set_title("Watershed Accuracies")
    plt.savefig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder", type=str)
    p.add_argument("-p", "--plot", action="store_true", default=False)
    args = p.parse_args()
    l, ca, wa = evaluate(args.model_folder)
    if args.plot:
        plot_histograms(l, ca, wa)


if __name__ == "__main__":
    model_folder = "/nemo/lab/vincentj/home/users/nicholb/organoid_data/model_unet"
    losses, classifier_accuracies, watershed_accuracies = evaluate(model_folder)
    plot_histograms(losses, classifier_accuracies, watershed_accuracies)
    print(f"Mean Loss - {np.mean(losses)}")
    print(f"Mean CAccuracy - {np.mean(classifier_accuracies)}")
    print(f"Mean WAccuracy - {np.mean(watershed_accuracies)}")