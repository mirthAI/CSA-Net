import torch
import numpy as np
import matplotlib.pyplot as plt


def save_visualization(outputs, label_batch, epoch_num, i_batch):
    """
    Processes the outputs and label batch, and saves visualization of the predictions and labels.

    Parameters:
    - outputs (torch.Tensor): The output predictions from the model.
    - label_batch (torch.Tensor): The batch of ground truth labels.
    - epoch_num (int): Current epoch number.
    - i_batch (int): Current batch index.

    Saves images to disk showing the predicted segmentation and actual labels.
    """
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1).squeeze(dim=1)
    rand_slice_out = outputs[0,:,:]
    rand_slice_out = rand_slice_out.cpu().detach().numpy()
    plt.imshow(rand_slice_out)
    plt.colorbar()
    path1 = f"./training_result/{epoch_num}_{i_batch}pred_image.png"
    plt.savefig(path1)
    plt.close()

    rand_label_slice = label_batch[0,:,:]
    rand_label_slice = rand_label_slice.cpu().detach().numpy()
    plt.imshow(rand_label_slice)
    plt.colorbar()
    path2 = f"./training_result/{epoch_num}_{i_batch}label_image.png"
    plt.savefig(path2)
    plt.close()