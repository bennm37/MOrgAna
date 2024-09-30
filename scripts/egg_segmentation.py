import numpy as np 
import matplotlib.pyplot as plt 
import skimage 
import skimage.io as io
from scipy.ndimage import binary_fill_holes



def curvature(i, threshold=0.35, opening=50, curve_thresh=0.023):
    img = io.imread(f"/Users/nicholb/Documents/data/egg_data/Td_embryo_dimension{str(i).zfill(2)}.tif")
    grayscale = skimage.color.rgb2gray(img)
    thresh = np.array(grayscale < threshold).astype(int)*255
    thresh = binary_fill_holes(thresh) 
    thresh = skimage.morphology.remove_small_objects(thresh, min_size=10000)
    footprint = skimage.morphology.disk(opening)
    thresh = skimage.morphology.binary_opening(thresh, footprint=footprint)
    contours = skimage.measure.find_contours(thresh, 0.5)[0]
    contours = contours[::10]
    curvature = np.zeros(len(contours))
    x = contours[:, 1]
    y = contours[:, 0]
    dx = np.gradient(x)
    ddx = np.gradient(dx)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx)/(dx**2 + dy**2)**1.5
    plt.title(f"Curvature of the contours of egg {i}")
    plt.imshow(img, cmap="gray")
    plt.scatter(contours[:, 1], contours[:, 0], linewidth=2, c=curvature>curve_thresh)
    plt.show()

for i in range(1, 50):
    curvature(i, curve_thresh=0.018)