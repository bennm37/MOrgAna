<img align="left" width="80" height="80" src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/morgana_icon.png" alt="morgana">

# MOrgAna

Welcome to MOrgAna (Machine-learning based Organoids Analysis) to segment and analyse 2D multi-channel images of organoids.

Optional: To use deep machine learning in generation of masks, please install the correct version of TensorFlow and cuDNN for your system:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows

## Using the software

This software is able to A) generate binary masks of organoids based on their bright-field images and with this mask, extract morphological information, generate a midline and a meshgrid. B) Provide analysis of fluorescence signals along the generated midline and enable quick and easy visual comparisons between conditions.

To download the software, run `pip install morgana` in terminal (MacOS) or command prompt(windows) followed by the command `python -m morgana`

<p align="center">
	<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/front_page.png" alt="front_page" width="350"/>
</p>

For advance python users looking to analyse multiple image folders at once, please refer to the jupyter notebook `morgana/Examples/MOrgAna_workflow_for_advance_python_users.ipynb'.

### A) Generate or Import Masks Tab
Each tif file in image folder should contain only one organoid with the brightfield channel as the starting image of each tif.

#### Creating binary masks

1. Manually create a `model` folder that contains a `trainingset` sub-folder. Select a few representative images (~5% of all images) and copy them into this sub-folder. If binary masks of this training set have already been created, place them in the same folder and name them as `..._GT.tif`. E.g. `20190509_01B_dmso.tif` and `20190509_01B_dmso_GT.tif`.

2. Run the segmentation app. Click `Specify model folder` and select the `model` folder created. If binary masks are missing, please manually annotate for each image by clicking on the image in the pop up window to create a boundary around your object of interest or right click on red dots to remove selection. 

<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/binary_mask.png" alt="binary_mask" width="400"/>
</p>
3. Select `Use Multi Layer Perceptrons` if Tensorflow and CUDA have been successfully installed and if you would like to use deep learning to generate additional binary masks. 

	Users can choose to adjust the following parameters of the model by clicking `Show/Hide params`
	* Sigmas: length scales (in pixels) used to generate the gaussian blurs of the input image
	* Downscaling: number of pixels used to resize the input image. This is mainly done to reduce  computation time, and a value of 500 is found to be enough in most applications.
	* Edge size: number of pixels used on the border of the mask to generate the edge of the organoid.
	* Pixel% extraction: percentage of pixels of the input image to be considered. 0: no pixels, 1: all pixels
	* Extraction bias: percentage of pixels extracted from the bright region of the mask. This parameter is useful when inputted gastruloids are particularly small and there is a huge bias in extracting background pixels.
	* Features: 'ilastik' or 'daisy'. In addition to the ilastik features (gaussian blur, laplacian of gaussian, difference of gaussian and gradient), daisy will compute many texture features from the inptu image. This gives more features to train on, but will slow down the training and prediction of new masks.

4. Once done, hit the `Train model` button. This may take some time :coffee:. Once completed, the message `##### Model saved!` will be seen on the terminal(MacOS) or command prompt(windows). If a model has previously been generated, select the model folder and the user can skip step 3 & 4 and jump to step 5. For models trained with Multi Layer Perceptrons, tick the option before selection of model folder.

5. To generate binary masks of new images, select the folder containing images in `Specify image folder` and click `Generate masks`. Once completed, the message `All images done!` will be displayed on the terminal(MacOS) or command prompt(windows). If you would like an overview of all masks generated, click on `Save overview image of masks` and save the pop-up image.

6. Click on `Inspect masks`. This will generate a overview of binary masks overlayed with their respective brightfield images. The mask generated with the watershed algorithm is shown in blue while the red mask is generated with the classifier algorithm.

<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/manual_selection_mask.png" alt="manual_selection_mask" width="800"/>
</p>

7. The other panel will allow the user to chose, for every image, the final mask type: 'ignore' (do not include selected image and mask), 'classifier' (red), 'watershed' (blue), 'manual' (manually create mask). Clicking `Show/Hide more parameters` will enable the user to change parameters such as downsampling, thinning and smoothing used in the generation of the final mask. Optional: select `Compute full meshgrid` to generate a meshgrid for straightening of organoid for later quantification. If disabled, meshgrid will automatically be generated later if required.

8. Next, `Compute all masks` will generate the final masks for every input image and save them into the `result_segmentation` subfolder. If 'manual' is selected, the user will be prompted to generate the manual mask on a separate window. As a rule of thumb, the classifier algorithm works most of the times. 


#### Import external masks

1. If binary masks of all images have already been generated, select `Import external masks`. This will reveal a new page. This feature allows import of images with multiple objects of interest.

<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/import_external_masks.png" alt="import_external_masks" width="350"/>
</p>

2. Specify image and mask folder with the `Specify image folder` and `Specify mask folder` buttons. Masks should be labeled as name of its respective image + file identifier. E.g. if the identifier is `_GT`: Image `20190509_01B_dmso.tif` and its mask `20190509_01B_dmso_GT.tif`.

3. Select `Include objects at border of images` if all partial images at edges of images are to be included. 

4. `Import Masks and Images` will create a mask and a image for each object detected in imported images and masks.


### B) Quantification

Click on the Quantification tab to enable morphological and fluorescence quantification with previously generated masks.
<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/quantification_tab.png" alt="quantification_tab" width="350"/>
</p>

1. Using the `Select new dataset` button, import all image folders previously generated or imported in the `Generate or Import Masks` tab into the preferred groups. Each group can refer to one condition or one timepoint. For groups spanning multiple timepoints, users may select the `Timelapse data` option. More groups can be created by clicking `Add New Group` at the top. If there is only one group, `Groups` can be disabled at the top after selection of dataset.

2. After importing all selected image folders, there are several options available below:
<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/extended_quantification_tab.png" alt="extended_quantification_tab" width="350"/>
</p>

* `Visualization quantification`: creates an overview of all meshgrids and composite images

* `Morphology quantification`: Analysis of the following morphological parameters calculated using the unprocessed mask (without straightening) or the straighted mask (straighted using the generated midline)
	* area
	* eccentricity
	* major_axis_length
	* minor_axis_length
	* equivalent_diameter
	* perimeter
	* euler_number
	* extent
	* orientation
	* locoefa_coeff (indication of complexity of shape)
	* `Use all parameters`: will display 10 graphs, each a quantification of the above parameters.
	
	Clicking `Visualize Morphological Parameter (s)` will display one or more of the following windows:

<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/area.png" alt="area" width="350"/>
</p>

	In this window, you can edit the quantification of morphological parameters by selecting the type of normalization and background subtraction. Users can also edit the graph shown by changing Pixel size/Scaler, Dimensionality, Plot type and Colormap with the options of removing groups, addition of legend or removal or raw data points on the graph. To view changes, click on `Apply Settings` after making the desired changes to options shown. `Compute statistics` shows P-values obtained from T-test, with the option of saving the p-values in a excel sheet. Users can also choose to save all resulting quantification values with the `Save Data as xlsx` button at the bottom. Square buttons at the top of the window can also be used to adjust the resulting graph with default options provided by matplotlib.


* `Fluorescence quantification`: Quantification of fluorescence in the chosen channel with respect to space with the selection of Antero-Posterior profile, Left-Right profile, Radial profile, Angular profile or simply with the average fluorescence intensity. `Compute graph` will display one such panel shown below:
<p align="center">
<img src="https://raw.githubusercontent.com/LabTrivedi/MOrgAna/master/morgana/Examples/app_screenshots/APprofile.png" alt="APprofile" width="350"/>
</p>
Users can choose to adjust method of quantification by changing Background subtraction type, Y axis normalization or selection of X axis normalization. If a spatial profile was chosen, the orientation of the profile can be signal-based. Users can similarly edit the colours of the graph with the Colormap, edit the X and Y axis labels, choose not to plot unwanted groups, include legends or remove raw data points from the graph shown. After altering the options, click on `Apply Settings` to view the changes. Default options of graphs by matplotib can also be changed with the square buttons at the top of the window.


## Supplementary information

Each subfolder containing the final masks also contains a segmentation_params.csv file generated during mask generation with the following information selected during creation of binary masks:
* filename
* chosen_mask: classifier (c), watershed (w), manual (m), ignore (i)
* down_shape
* thinning
* smoothing

All morphological properties of organoids are computed when required and saved as `..._morpho_params.json` into the same subfolder as the final masks (`result_segmentation`)

These include:
* 'input_file'
* 'mask_file'
* 'centroid'
* 'slice'
* 'area'
* 'eccentricity' (perfect circle:0, elongated ellipse:~1)
* 'major_axis_length'
* 'minor_axis_length'
* 'equivalent_diameter'
* 'perimeter'
* 'anchor_points_midline'
* 'N_points_midline'
* 'x_tup'
* 'y_tup'
* 'midline'
* 'tangent'
* 'meshgrid_width'
* 'meshgrid'

The `_morpho_straight_params.json` is computed when required and saved into the same subfolder as the final masks (`result_segmentation`). It contains the following infomation:
* area
* eccentricity
* major_axis_length
* minor_axis_length
* equivalent_diameter
* perimeter
* euler_number
* extent
* orientation
* locoefa_coeff (indication of complexity of shape)


The average fluorescence intensities, and those along the Antero-Posterior, Left-Right, Radial and Angular profile of organoids are computed when required and saved as `..._fluo_intensity.json` into the same subfolder as the final masks (`result_segmentation`).

