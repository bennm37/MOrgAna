from morgana.MLModel import predict, io as ioML
import morgana.DatasetTools.io as ioDT
from skimage.io import imread
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
    pred, prob = predict.predict_image_unet(img, scaler, model, image_size=(512, 512))
    mask_pred = predict.make_mask(pred)
    edge_prob = ((2**16 - 1) * prob[:, :, 2]).astype(np.uint16)
    watershed = predict.make_watershed(mask_pred, edge_prob)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    # ax[1].imshow(mask_pred)
    ax[1].imshow(watershed)
    plt.show()
