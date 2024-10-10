import os
import tqdm
import numpy as np
import matplotlib as mpl
from textwrap import wrap
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import find_boundaries
from matplotlib import rc
import cv2
from morgana.DatasetTools import io

rc("pdf", fonttype=42)


def load_masks(input_folder, start=None, stop=None, downshape=1):
    flist_in = io.get_image_list(input_folder)
    segment_folder = os.path.join(input_folder, "result_segmentation")
    flist_ws = io.get_image_list(segment_folder, "_watershed.tif", "include")
    flist_cl = io.get_image_list(segment_folder, "_classifier.tif", "include")
    if start is None:
        start = 0
    if stop is None:
        stop = len(flist_in)
    flist_in = flist_in[start:stop]
    flist_ws = flist_ws[start:stop]
    flist_cl = flist_cl[start:stop]

    n_img = len(flist_in)
    imgs = [0.0 for i in range(n_img)]
    classifiers = [0.0 for i in range(n_img)]
    watersheds = [0.0 for i in range(n_img)]
    manuals = [None for i in range(n_img)]
    for i in tqdm.tqdm(range(n_img)):
        img = imread(flist_in[i]).astype(float)
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        imgs[i] = img[0, ::downshape, ::downshape]
        classifiers[i] = imread(flist_cl[i])[::downshape, ::downshape].astype(float)
        watersheds[i] = imread(flist_ws[i])[::downshape, ::downshape].astype(float)
        name = os.path.split(flist_in[i])[-1]
        manual_path = flist_in[i].replace(name, f"result_segmentation/{name[:-4]}_manual.tif")
        if os.path.exists(manual_path):
            manuals[i] = imread(manual_path)[::downshape, ::downshape].astype(float)
    return imgs, classifiers, watersheds, manuals


def alpha_overlay(img, overlay):
    alpha_overlay = overlay[:, :, 3] / 255.0
    for color in range(0, 3):
        img[:, :, color] = alpha_overlay * overlay[:, :, color] + img[:, :, color] * (1 - alpha_overlay)
    return img


def create_icons(
    imgs, classifiers, watersheds, manuals, cc=[255, 0, 0], wc=[0, 255, 255], mc=[255, 255, 0], size=200
):
    cc, wc = np.array(cc), np.array(wc)
    n_img = len(imgs)
    icons = [0.0 for i in range(n_img)]
    for i in range(n_img):
        if np.max(imgs[i]) > 255:
            img = np.array(imgs[i] / 256.0)  # assumes 16 bit
        else:
            img = np.array(imgs[i])
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        vmin, vmax = np.percentile(img, 1.0), np.percentile(img, 99.0)
        img = np.clip(img, vmin, vmax)
        img = 255 * (img - vmin) / (vmax - vmin)
        img = img.astype(np.uint8)
        classifier = np.where(classifiers[i] > 0, 1, 0).astype(np.uint8)
        watershed = watersheds[i]
        img = cv2.resize(img, (size, size))
        classifier = cv2.resize(classifier, (size, size))
        watershed = cv2.resize(watershed, (size, size))
        icon = img.copy()
        w_boundaries = find_boundaries(watershed > 0, mode="inner")
        icon[w_boundaries] = wc[:3]
        c_boundaries = find_boundaries(classifier > 0, mode="inner")
        icon[c_boundaries] = cc[:3]
        if manuals[i] is not None:
            manual = manuals[i]
            manual = cv2.resize(manual, (size, size))
            m_boundaries = find_boundaries(manual > 0, mode="inner")
            icon[m_boundaries] = mc[:3]
        icons[i] = icon
    return icons


def generate_overview(input_folder, saveFig=True, fileName="", start=None, stop=None, downshape=1):
    print("Generating recap image at", input_folder)
    imgs, classifiers, watersheds = load_masks(input_folder, start, stop, downshape)
    flist_in = io.get_image_list(input_folder)[start:stop]
    n_img = len(flist_in)
    ncols = 5
    nrows = (n_img - 1) // 5 + 1
    fig, ax = plt.subplots(figsize=(3 * ncols, 3 * nrows), nrows=nrows, ncols=ncols)
    ax = ax.flatten()
    for i in tqdm.tqdm(range(n_img)):
        _, filename = os.path.split(flist_in[i])
        filename, _ = os.path.splitext(filename)
        ax[i].imshow(
            imgs[i],
            "gray",
            interpolation="none",
            vmin=np.percentile(imgs[0], 1.0),
            vmax=np.percentile(imgs[0], 99.0),
        )
        cmap = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", ["black", "red"], 256)
        ax[i].imshow(classifiers[i], cmap=cmap, interpolation="none", alpha=0.4)
        cmap = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", ["black", "aqua"], 256)
        ax[i].imshow(watersheds[i], cmap=cmap, interpolation="none", alpha=0.3)
        ax[i].set_title("\n".join(wrap(filename, 20)), fontsize=8)
    for a in ax:
        a.axis("off")
    for j in range(i + 1, len(ax)):
        ax[j].remove()
    if saveFig:
        print("Saving image...")
        # save figure
        _, cond = os.path.split(input_folder)
        print(fileName)
        if fileName == "":
            fileName = os.path.join(
                input_folder,
                "result_segmentation",
                cond + "_recap_classifier.png",
            )
        fig.savefig(fileName, dpi=300)
        print("Done saving!")
    return fig
