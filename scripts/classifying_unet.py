from morgana.MLModel import predict, io as ioML
import morgana.DatasetTools.io as ioDT
from skimage.io import imread
import numpy as np 
import matplotlib.pyplot as plt

model_folder = "/Users/nicholb/Documents/data/organoid_data/240924_model/model_copy"
image_folder = f"{model_folder}/data"
flist_in = ioDT.get_image_list(image_folder)
model, scaler, params = ioML.load_model(model_folder, deep=True)
img = imread(flist_in[25])
predicted = predict.predict_image_unet(img, scaler, model, image_size=(512,512))
predicted = np.array(predicted)[0]
gray_predicted = np.argmax(predicted, axis=-1)
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(img)
ax[0].imshow(gray_predicted, alpha=0.5)
ax[1].imshow(predicted[:,:,0])
plt.show()