import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import SimpleITK as sitk
from PIL import Image
import numpy as np
import cv2

def random_horizontal_flip(image,next_image, prev_image, segmentation):
    # Generate a random number to decide whether to flip or not
    flip = random.choice([True, False])
    
    # Perform horizontal flipping if flip is True
    if flip:
        flipped_image = np.fliplr(image)
        flipped_next_image = np.fliplr(next_image)
        flipped_prev_image = np.fliplr(prev_image)
        flipped_segmentation = np.fliplr(segmentation)
    else:
        flipped_image = image
        flipped_next_image = next_image
        flipped_prev_image = prev_image
        flipped_segmentation = segmentation
    
    return flipped_image,flipped_next_image,flipped_prev_image,flipped_segmentation



class RandomGenerator(object):
    """
    Applies random transformations to a sample including horizontal flips and resizing to a target size.

    Parameters:
        output_size (tuple): Desired output dimensions (height, width) for the images and labels.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # Unpack the sample dictionary to individual components
        image, label = sample['image'], sample['label']
        next_image, prev_image = sample['next_image'], sample['prev_image']
        
        # Apply a random horizontal flip to the images and label
        image,next_image, prev_image, label = random_horizontal_flip(image, next_image, prev_image, label)
        # Check if the current size matches the desired output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # Rescale images to match the specified output size using cubic interpolation
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            next_image = zoom(next_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            prev_image = zoom(prev_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            # Rescale the label using nearest neighbor interpolation (order=0) to avoid creating new labels
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert numpy arrays to PyTorch tensors and add a channel dimension to images
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        next_image = torch.from_numpy(next_image.astype(np.float32)).unsqueeze(0)
        prev_image = torch.from_numpy(prev_image.astype(np.float32)).unsqueeze(0)
        # Return the modified sample as a dictionary
        sample = {'image': image, 'next_image': next_image, 'prev_image': prev_image, 'label': label.long()}
        return sample


def extract_and_increase_number(file_name):
    """
    Generates the filenames for the next and previous sequence by incrementing and decrementing the numerical part of a given filename.

    Parameters:
        file_name (str): The original filename from which to derive the next and previous filenames. 
                         The filename must end with a numerical value preceded by an underscore.

    Returns:
        tuple: Contains two strings, the first being the next filename in sequence and the second 
               the previous filename in sequence. If the original number is 0, the previous filename 
               will also use 0 to avoid negative numbering.
    """
    parts = file_name.rsplit("_", 1)
    parts_next = parts[0]
    parts_prev = parts[0]
    number = int(parts[1])
    
    next_number = number + 1
    prev_number = number - 1
    if prev_number== -1:
        pre_number = 0
    
    next_numbers = str(next_number)
    prev_numbers = str(prev_number)
    next_file_name = parts_next+"_"+str(next_numbers)
    prev_file_name = parts_prev+"_"+str(prev_numbers)

    return next_file_name,prev_file_name    
    
    
    
def check_and_create_file(file_name, image_name, folder_path):
    file_path = os.path.join(folder_path, "trainingImages", file_name+'.npy')
    if os.path.exists(file_path):
        return file_name
    else:
        available_name = image_name
        return available_name 


class CSANet_dataset(Dataset):
    """
    Dataset handler for CSANet, designed to manage image and mask data for training and testing phases.

    Attributes:
        base_dir (str): Directory where image and mask data are stored.
        list_dir (str): Directory where the lists of data splits are located.
        split (str): The current dataset split, indicating training or testing phase.
        transform (callable, optional): A function/transform to apply to the samples.

    Note:
        This class expects directory structures and file naming conventions that match the specifics
        given in the initialization arguments.
    """
    
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.image_sample_list = open(os.path.join(list_dir, 'train_image.txt')).readlines()
        self.mask_sample_list = open(os.path.join(list_dir, 'train_mask.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_image" or self.split == "train_image_train" or self.split == "train_image_test":
            
            slice_name = self.image_sample_list[idx].strip('\n')
            image_data_path = os.path.join(self.data_dir, "trainingImages", slice_name+'.npy')
            image = np.load(image_data_path)
            #print("##################################### image path = ", image_data_path)
            # Manage sequence continuity by fetching adjacent slices
            next_file_name, prev_file_name = extract_and_increase_number(slice_name)
            
            next_file_name = check_and_create_file (next_file_name, slice_name, self.data_dir)
            prev_file_name = check_and_create_file (prev_file_name, slice_name, self.data_dir)
            
            
            next_image_path = os.path.join(self.data_dir, "trainingImages", next_file_name +'.npy')
            prev_image_path = os.path.join(self.data_dir, "trainingImages", prev_file_name +'.npy')
            
            next_image = np.load(next_image_path)
            prev_image = np.load(prev_image_path)           
            
            
            mask_name = self.mask_sample_list[idx].strip('\n')
            label_data_path = os.path.join(self.data_dir, "trainingMasks", mask_name+'.npy')
            #print("############################################# label path = ", label_data_path)
            label = np.load(label_data_path)
            
            sample = {'image': image, 'next_image': next_image, 'prev_image': prev_image, 'label': label}
        
            if self.transform:
                sample = self.transform(sample) # Apply transformations if specified
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
        else:
            # Handling testing data, assuming single volume processing
            vol_name = self.sample_list[idx].strip('\n')
            image_data_path = os.path.join(self.data_dir, "testVol", vol_name)
            label_data_path = os.path.join(self.data_dir, "testMask", vol_name)
            
            image_new = sitk.ReadImage(image_data_path) 
            img = sitk.GetArrayFromImage(image_new)
            
            
            next_image = sitk.GetArrayFromImage(image_new).astype(np.float64)
            prev_image = sitk.GetArrayFromImage(image_new).astype(np.float64)
            
            # Preprocess image data for testing phase
            combined_slices = sitk.GetArrayFromImage(image_new).astype(np.float64)
            
            
            for i in range(img.shape[0]):
                img_array = img[i, :, :].astype(np.uint8)
                p1 = np.percentile(img_array, 1)
                p99 = np.percentile(img_array, 99)

                normalized_img = (img_array - p1) / (p99 - p1)
                normalized_img = np.clip(normalized_img, 0, 1)

                combined_slices[i,:,:] = normalized_img
                
                if i-1 > -1 :
                    next_image[i-1,:,:] = combined_slices[i,:,:]
                
                if i-1<0:
                    prev_image[i,:,:] = combined_slices[i,:,:]
                else :
                    prev_image[i,:,:] = combined_slices[i-1,:,:]
            
            next_image[img.shape[0]-1,:,:] = combined_slices[img.shape[0]-1,:,:]
            
            segmentation = sitk.ReadImage(label_data_path)
            label = sitk.GetArrayFromImage(segmentation)
            sample = {'image': combined_slices, 'next_image': next_image, 'prev_image': prev_image, 'label': label}
            if self.transform:
                sample = self.transform(sample) # Apply transformations if specified
            num_string = self.sample_list[idx].strip('\n')
            case_num = num_string.split('.')[0]
            sample['case_name'] = case_num
            return sample

        
