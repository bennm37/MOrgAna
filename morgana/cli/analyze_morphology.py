import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes, closing, disk
from skimage.segmentation import watershed
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import erosion
from scipy import ndimage as ndi
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm


# Function to fill holes in the binary mask
def fill_holes(mask, hole_size_threshold=500):
    # Remove small holes by specifying a size threshold
    filled_mask = remove_small_holes(mask.astype(bool), area_threshold=hole_size_threshold)
    return filled_mask.astype(np.uint8)


# Function to apply morphological closing to the mask
def apply_closing(mask, disk_size=5):
    # Apply morphological closing to the mask using a disk-shaped structuring element
    closed_mask = closing(mask, disk(disk_size))
    return closed_mask.astype(np.uint8)


def erode_mask(mask, erosion_size=5):
    # Create a disk-shaped structuring element with a radius of 5 pixels
    selem = disk(erosion_size)
    eroded_mask = erosion(mask, selem)
    return eroded_mask


# Function to apply watershed segmentation
def apply_watershed(image, mask):
    # Use distance transform on the mask
    distance = ndi.distance_transform_edt(mask)

    # Find local maxima as markers for watershed
    local_maxi = ndi.label(distance == np.max(distance))[0]

    # Apply watershed algorithm
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask)

    return labels


def extract_morphological_properties(segmented_image):
    labeled_image = label(segmented_image)
    properties = regionprops(labeled_image)

    # Initialize an empty list to store region properties
    prop_list = []

    for i, prop in enumerate(properties):
        area = prop.area
        perimeter = prop.perimeter
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        eccentricity = prop.eccentricity
        solidity = prop.solidity

        # Append the properties for each region to the list
        prop_list.append(
            {
                "Region": i + 1,
                "Area": area,
                "Perimeter": perimeter,
                "Circularity": circularity,
                "Eccentricity": eccentricity,
                "Solidity": solidity,
            }
        )

    # Convert the list of properties to a Pandas DataFrame
    df = pd.DataFrame(prop_list)

    return df


# Function to process the image
def process_image(image):
    # Create a mask where pixel value is 1 or 2
    mask = np.logical_or(image == 1, image == 2).astype(np.uint8)

    # Fill larger holes inside the mask with a threshold for hole size
    filled_mask = fill_holes(mask, hole_size_threshold=10000)

    # Apply morphological closing to the filled mask
    closed_mask = apply_closing(filled_mask, disk_size=10)

    eroded_mask = erode_mask(closed_mask, erosion_size=5)

    # Perform watershed segmentation
    segmented_image = apply_watershed(image, eroded_mask)

    return eroded_mask, segmented_image


# Function to display original and processed images
def show_images(original, filled_mask, segmented_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(filled_mask, cmap="gray")
    ax[1].set_title("Filled Mask")
    ax[1].axis("off")

    ax[2].imshow(segmented_image, cmap="nipy_spectral")
    ax[2].set_title("Watershed Segmentation")
    ax[2].axis("off")

    plt.show()


def running_morphological_on_single_image(image_path):
    image = imread(image_path)

    # Process the image
    filled_mask, segmented_image = process_image(image)

    # Extract and display morphological properties
    df = extract_morphological_properties(segmented_image)

    # print(df.head)
    # Define the output file path by replacing "classifier" with "segmented"
    output_image_path = image_path.replace("classifier", "segmented")

    # Save the segmented image
    imsave(output_image_path, segmented_image.astype(np.uint8))

    # Display the original and processed images
    # show_images(image, filled_mask, segmented_image)

    return df


# Function to process each ROI folder (this will be run in parallel)
def process_roi(args):
    folder, roi_folder, result_segmentation_folder = args
    roi_combined_df = pd.DataFrame()

    # Look for all image files containing 'classifier' in the result_segmentation folder
    for image_file in os.listdir(result_segmentation_folder):
        if "classifier" in image_file and image_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            image_path = os.path.join(result_segmentation_folder, image_file)

            # Run the function on each image
            df = running_morphological_on_single_image(image_path)

            # Add extra columns for folder name and ROI folder name
            df["MainFolder"] = folder
            df["ROIFolder"] = roi_folder

            # Append to the ROI DataFrame
            roi_combined_df = pd.concat([roi_combined_df, df], ignore_index=True)

    # Save the individual CSV for this ROI
    # output_csv_path = os.path.join(result_segmentation_folder, f"{roi_folder}_morphological_results.csv")
    # roi_combined_df.to_csv(output_csv_path, index=False)

    return roi_combined_df  # Return the DataFrame for further combination


# Main function to iterate through the folders and set up multiprocessing
def process_folder_structure_in_parallel(parent_folder):
    combined_df = pd.DataFrame()  # DataFrame to hold all results
    roi_tasks = []  # List to store tasks for multiprocessing

    # Iterate through each folder in the parent folder (e.g., folder1image1, folder2image2)
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)

        if os.path.isdir(folder_path):  # Check if it is a folder
            # Iterate through ROI subfolders (e.g., ROI1, ROI2)
            for roi_folder in os.listdir(folder_path):
                roi_folder_path = os.path.join(folder_path, roi_folder)

                if os.path.isdir(roi_folder_path):  # Check if it is a folder
                    # Look for result_segmentation folder
                    result_segmentation_folder = os.path.join(roi_folder_path, "result_segmentation")

                    if os.path.exists(result_segmentation_folder):
                        # Prepare the arguments for each ROI to be processed in parallel
                        roi_tasks.append((folder, roi_folder, result_segmentation_folder))

    # Use multiprocessing with Pool to process each ROI folder in parallel
    with Pool() as pool:
        # tqdm for progress tracking
        results = list(tqdm(pool.imap(process_roi, roi_tasks), total=len(roi_tasks)))

    # Combine all the results from individual ROIs into a single DataFrame
    for df in results:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


# Main code to run the process
if __name__ == "__main__":
    # Input: the parent folder containing folder1image1, folder2image2, etc.
    parent_folder = "/Users/perezg/Documents/data/2024/240924_organo_segment/241002_small/"

    # Process the folder structure in parallel
    combined_df = process_folder_structure_in_parallel(parent_folder)

    # Save the combined DataFrame to a CSV file
    combined_output_path = os.path.join(parent_folder, "combined_morphological_results.csv")
    combined_df.to_csv(combined_output_path, index=False)

    print(f"Combined results saved to {combined_output_path}")
