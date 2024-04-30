import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import csv
import re
import cv2
def ensure_directory_exists(path):
    """
    Ensures that a directory exists; if not, creates it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def ensure_file_exists(file_path):
    """
    Ensures that a file exists; if not, creates an empty file.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("") 
        print(f"Created file: {file_path}")
    else:
        print(f"File already exists: {file_path}")

# Directories and paths setup
vol_dir = "data/trainVol"
mask_dir = "data/trainMask"
path_to_scan = os.path.join(os.getcwd(), vol_dir)
mask_path_to_scan = os.path.join(os.getcwd(), mask_dir)

save_image_path = "data/train_npz/trainingImages"
save_mask_path = "data/train_npz/trainingMasks"
ensure_directory_exists(save_image_path)
ensure_directory_exists(save_mask_path)

# Process volume files
for file in os.listdir(path_to_scan):
    if not file.startswith('.'):
        vol_path = os.path.join(path_to_scan, file)
        image = sitk.ReadImage(vol_path)
        img = sitk.GetArrayFromImage(image)
        number = file.split('.')[0]
        for i in range(img.shape[0]):
            img_array = img[i, :, :].astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            p1 = np.percentile(img_array, 1)
            p99 = np.percentile(img_array, 99)
    
            normalized_img = (img_array - p1) / (p99 - p1)
            normalized_img = np.clip(normalized_img, 0, 1)
            slice_name = f"{number}_vol_slice_{i}.npy"
            slice_path = os.path.join(save_image_path, slice_name)
            np.save(slice_path, normalized_img)

# Process mask files
for file in os.listdir(mask_path_to_scan):
    if not file.startswith('._'):
        mask_path = os.path.join(mask_path_to_scan, file)
        mask = sitk.ReadImage(mask_path)
        mask_img = sitk.GetArrayFromImage(mask)
        number = file.split('.')[0]
        for i in range(mask_img.shape[0]):
            slice_name = f"{number}_mask_slice_{i}.npy"
            slice_path = os.path.join(save_mask_path, slice_name)
            np.save(slice_path, mask_img[i,:,:])

# Create CSV file for training data
if not os.path.exists("CSANet/lists"):
        os.makedirs("CSANet/lists")
csv_filename = "CSANet/lists/train.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image', 'mask'])
    for file in os.listdir(path_to_scan):
        if not file.startswith('._'):
            number = file.split('.')[0]
            for i in range(img.shape[0]):
                seg_file_name = f"{number}_mask_slice_{i}.npy"
                vol_file_name = f"{number}_vol_slice_{i}.npy"
                csv_writer.writerow([vol_file_name, seg_file_name])

# Ensure essential files exist
image_list_path = "CSANet/lists/train_image.txt"
mask_list_path = "CSANet/lists/train_mask.txt"
vol_list_path = "CSANet/lists/test_vol.txt"
ensure_file_exists(image_list_path)
ensure_file_exists(mask_list_path)
ensure_file_exists(vol_list_path)

# Generate file lists for image and mask
def generate_file_list(data, key, file_path):
    num = data[key].values.size
    names = [data[key].values[i].split('.')[0] for i in range(num)]
    with open(file_path, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

# Creating file lists
data = pd.read_csv(csv_filename)
generate_file_list(data, 'image', image_list_path)
generate_file_list(data, 'mask', mask_list_path)

# List files for test volumes
folder_path = 'data/testVol'
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
with open(vol_list_path, 'w') as file:
    for file_name in files:
        if not file_name.startswith('._'):
          file.write(file_name + '\n')
