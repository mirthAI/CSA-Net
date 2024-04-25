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

class DiceLoss(nn.Module):
    """
    Implements a Dice loss for evaluating segmentation performance, where Dice loss is a measure of overlap 
    between two samples and can be used as a loss function for training deep learning models for segmentation tasks.

    Attributes:
    - n_classes (int): Number of classes for segmentation.
    """
    def __init__(self, n_classes):
        """
        Initializes the DiceLoss module with the number of classes.

        Parameters:
        - n_classes (int): Number of segmentation classes.
        """
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """
        Converts a tensor of indices of a categorical variable into a one-hot encoded format.

        Parameters:
        - input_tensor (torch.Tensor): Tensor containing indices that will be one-hot encoded.

        Returns:
        - torch.Tensor: One-hot encoded tensor.
        """
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """
        Computes the Dice loss between the predicted scores and the one-hot encoded target.

        Parameters:
        - score (torch.Tensor): Predicted scores for each class.
        - target (torch.Tensor): One-hot encoded true labels.

        Returns:
        - float: Dice loss value.
        """
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Forward pass for calculating Dice loss for multiple classes.

        Parameters:
        - inputs (torch.Tensor): Input logits or softmax predictions.
        - target (torch.Tensor): Ground truth labels.
        - weight (list of float, optional): Class weights.
        - softmax (bool, optional): Whether to apply softmax to inputs before calculating loss.

        Returns:
        - float: Mean Dice loss across all classes.
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice 
        return loss / (self.n_classes - 1)