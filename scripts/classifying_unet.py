from morgana.MLModel import predict, io as ioML
import morgana.DatasetTools.io as ioDT
from skimage.io import imread
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
import matplotlib.pyplot as plt

model_folder = "/Users/nicholb/Documents/data/organoid_data/fullModel"
# model_folder = "/Users/nicholb/Documents/data/organoid_data/240924_model/model_copy"
image_folder = (
    "/Users/nicholb/Documents/data/organoid_data/seperatedStacks/image_1_MMStack_control_DMSO_1-1.ome_restacked/ROI2"
)
flist_in = ioDT.get_image_list(image_folder, string_filter="_GT", mode_filter="exclude")
model, scaler, params = ioML.load_model(model_folder, model="MLP")
for f in flist_in:
    img = imread(f)
    predicted = predict.predict_image_unet(img, scaler, model, image_size=(512, 512))
    predicted = np.array(predicted)[0]
    negative = ndi.binary_fill_holes(pred == 0)
    mask_pred = (pred == 1) * negative
    edge_prob = ((2**16 - 1) * prob[2]).astype(np.uint16)
    mask_pred = mask_pred.astype(np.uint8)
    gray_predicted = np.argmax(predicted, axis=-1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    gray = np.argmax(predicted * np.array([[[0.1, 1, 10]]]), axis=-1)
    interior = gray == 1
    gray[remove_small_objects(remove_small_holes(interior, 100), 500)] = 1
    edge = gray == 2
    ax[1].imshow(gray)
    plt.show()
