import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
import nibabel as nib
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
import math


def computing_COM_distance(mask_array, pred_array, US_indicies, spacing):
    
    """
    Computes the average physical distance between the centers of mass (COM) of corresponding predicted and ground truth masks.

    Parameters:
    - mask_array (np.array): Array of ground truth masks.
    - pred_array (np.array): Array of predicted masks.
    - US_indicies (list of int): Indices of the mask slices to be processed.
    - spacing (tuple of float): Physical spacing between pixels in the masks.

    Returns:
    - float: Mean physical distance between centers of mass across specified mask slices.
    """
    
    dist = []
    for num, US_num in enumerate(US_indicies):
        predicted_mask = pred_array[US_num].astype('uint8')
        ground_truth_mask = mask_array[US_num].astype('uint8')
        cy_hist, cx_hist = ndi.center_of_mass(predicted_mask)
        cy_us, cx_us = ndi.center_of_mass(ground_truth_mask)
        temp = math.dist([cx_hist, cy_hist], [cx_us, cy_us])
        phy_temp = temp * spacing[0]
        dist.append(phy_temp)
    distances = np.array(dist)
    
    if distances.size > 0:
        percentile_95 = np.percentile(distances, 95)
        mean_of_95th_percentile = np.mean(distances[distances <= percentile_95])
    else:
        percentile_95 = 0
        mean_of_95th_percentile = 0
    return mean_of_95th_percentile
      



def Dice_cal(image1, image2):
    
    """
    Calculates Dice coefficients for multiple class labels between two images.

    Parameters:
    - image1 (SimpleITK.Image): First image for comparison.
    - image2 (SimpleITK.Image): Second image for comparison.

    Returns:
    - tuple: Dice coefficients for each class label (1, 2, 3, 4).
    """
    
    class_labels = [1 , 2, 3, 4]
    for num_labels in class_labels:
        # Create binary masks for each class
        mask1 = sitk.Cast(image1 == num_labels, sitk.sitkInt32)
        mask2 = sitk.Cast(image2 == num_labels, sitk.sitkInt32)
        
        overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_filter.Execute(mask1, mask2)
        
        if num_labels == 1:
            dice_coeff_1 = overlap_filter.GetDiceCoefficient()
        elif num_labels == 2:
            dice_coeff_2 = overlap_filter.GetDiceCoefficient()
        elif num_labels == 3:
            dice_coeff_3 = overlap_filter.GetDiceCoefficient()
        elif num_labels == 4:
            dice_coeff_4 = overlap_filter.GetDiceCoefficient()
            
    return dice_coeff_1, dice_coeff_2, dice_coeff_3, dice_coeff_4

def compute_class_hausdorff(labels, outputs, class_index, spacing):
    
    """
    Computes the Hausdorff distance for a specific class based on its segmentation masks.

    Parameters:
    - labels (np.array): Array of ground truth labels for all classes.
    - outputs (np.array): Array of predicted labels for all classes.
    - class_index (int): Index of the class for which to compute the distance.
    - spacing (tuple of float): Physical spacing of the images.

    Returns:
    - float: Computed Hausdorff distance for the specified class.
    """
    
    US_indicies = []
    new_labels = labels[:,:,:,class_index]
    new_outputs = outputs[:,:,:,class_index]
    
    for z in range(new_labels.shape[0]):
        if np.sum(new_labels[z]) > 0 and np.sum(new_outputs[z]) > 0:
            US_indicies.append(z)
    
    hausdorff_dist = computing_COM_distance(new_labels, new_outputs, US_indicies, spacing)
    return hausdorff_dist


def test_single_volume(image_next, image, image_prev, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    
    """
    Tests a single volume for segmentation using a deep learning model, and evaluates segmentation accuracy using Dice and Hausdorff distances.

    Parameters:
    - image_next, image_prev, image (np.array): Current, previous, and next slices of the image volume.
    - label (np.array): Ground truth labels for the current image slice.
    - net (torch.nn.Module): Neural network model used for segmentation.
    - classes (int): Number of segmentation classes.
    - patch_size (list of int): Size of the patches processed by the network.
    - test_save_path (str, optional): Path to save the segmentation results.
    - case (str, optional): Identifier for the case being tested.

    Returns:
    - tuple: Dice coefficients and Hausdorff distances for each class.
    """
    # Convert tensors to numpy arrays for processing
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image_next, image_prev = image_next.squeeze(0).cpu().detach().numpy(), image_prev.squeeze(0).cpu().detach().numpy()
    
    # Initialize prediction array
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            # Resize slices if necessary to match the network's expected input size
            slice = image[ind, :, :]
            slice_prev = image_prev[ind, :, :]
            slice_next = image_next[ind, :, :]
            
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                slice_prev = zoom(slice_prev, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice_next = zoom(slice_next, (patch_size[0] / x, patch_size[1] / y), order=3)
            
            # Convert slices to tensors and run through the network
            input_curr = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            input_prev = torch.from_numpy(slice_prev).unsqueeze(0).unsqueeze(0).float().cuda()
            input_next = torch.from_numpy(slice_next).unsqueeze(0).unsqueeze(0).float().cuda()
            
            net.eval()
            
            with torch.no_grad():
                outputs = net(input_prev, input_curr , input_next)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        # Handle single slice case
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

            
    
    
    # Prepare data for analysis and visualization
    Result_path = "./Result"
    if not os.path.exists(Result_path):
        os.makedirs(Result_path)
    num_case = case[0]
    test_vol_path = "../data/testVol" + "/" + num_case + ".nii.gz"
    vol_image = sitk.ReadImage(test_vol_path)
    
    # Create SimpleITK images from numpy arrays for evaluation
    pred_image = sitk.GetImageFromArray(prediction)
    pred_image.SetSpacing(vol_image.GetSpacing())
    pred_image.SetDirection(vol_image.GetDirection())
    pred_image.SetOrigin(vol_image.GetOrigin())
    pred_path = './Result/' + num_case +'_segmentation.nii.gz' 
    sitk.WriteImage(pred_image, pred_path)
    
    label_image = sitk.GetImageFromArray(label)
    label_image.SetSpacing(vol_image.GetSpacing())
    label_image.SetDirection(vol_image.GetDirection())
    label_image.SetOrigin(vol_image.GetOrigin())
    label_path = './Result/' + num_case +'_label_segmentation.nii.gz' 
    sitk.WriteImage(label_image, label_path)
    
    # Load ground truth mask for evaluation
    mask_path = "../data/testMask/"+num_case+".nii.gz"
    mask_img  = sitk.ReadImage(mask_path)
    
    image1 = pred_image
    image2 = mask_img
    dice_coeff_1, dice_coeff_2, dice_coeff_3, dice_coeff_4 = 0.0, 0.0, 0.0, 0.0
    
    # Dice Coefficient Calculation
    dice_coeff_1, dice_coeff_2, dice_coeff_3, dice_coeff_4= Dice_cal(image1, image2)

    
    labels = np.eye(classes)[label]
    outputs = np.eye(classes)[prediction]
    spacing = vol_image.GetSpacing() 
    
    # Hausdroff Distance Calculation
    hausdorff_dist_1 = compute_class_hausdorff(labels, outputs, 1, spacing)
    hausdorff_dist_2 = compute_class_hausdorff(labels, outputs, 2, spacing)
    hausdorff_dist_3 = compute_class_hausdorff(labels, outputs, 3, spacing)
    hausdorff_dist_4 = compute_class_hausdorff(labels, outputs, 4, spacing)
    
    return dice_coeff_1, dice_coeff_2, dice_coeff_3, dice_coeff_4, hausdorff_dist_1,hausdorff_dist_2, hausdorff_dist_3, hausdorff_dist_4
