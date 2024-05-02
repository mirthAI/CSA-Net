import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
import time
import torch.nn as nn
import torch.optim as optim
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_CSANet import CSANet_dataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils import test_single_volume
from datasets.dataset_CSANet import CSANet_dataset, RandomGenerator


"""
This script configures and initializes training for the CSANet segmentation model using Vision Transformers. It handles command-line arguments for various training parameters, sets up deterministic options for reproducibility, and initializes the model with specified configurations.

Parameters:
- volume_path: Directory for validation volume data.
- dataset: Name of the dataset or experiment.
- num_classes: Number of output classes for segmentation.
- list_dir: Directory containing lists of data samples.
- max_iterations: Maximum number of iterations to train.
- max_epochs: Maximum number of epochs to train.
- batch_size: Number of samples per batch.
- seed: Seed for random number generators for reproducibility.
- n_gpu: Number of GPUs to use.
- img_size: Size of the input images.
- base_lr: Base learning rate for the optimizer.
- deterministic: Flag to set training as deterministic.
- n_skip: Number of skip connections in the model.
- vit_name: Name of the Vision Transformer model configuration.
- vit_patches_size: Size of patches for the ViT model.

The script also loads and validates the model from a saved state if available and performs inference to evaluate the model on a test dataset.
"""

# Setup command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data', help='root dir for validation volume data') 
parser.add_argument('--dataset', type=str,
                    default='CSANet', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--max_iterations', type=int,
                    default=300000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()




def vol_inference(args, model, test_save_path=None, validation=False):
    """
    Performs inference on a test dataset, computes performance metrics such as Dice coefficients and distances, 
    and can operate in validation mode or test mode based on a flag.

    Parameters:
    - args (Namespace): Contains all the necessary settings such as dataset paths, number of classes, image size, etc.
    - model (torch.nn.Module): The trained model to be evaluated.
    - test_save_path (str, optional): Path where test outputs (such as images) can be saved.
    - validation (bool, optional): If True, function returns average Dice coefficient for validation purposes.
                                   If False, returns a string message indicating test completion.

    Returns:
    - float: If validation is True, returns the average Dice coefficient.
    - str: If validation is False, returns a completion message "Testing Finished!"

    The function logs the number of test iterations, processes each test sample to compute Dice coefficients and distances,
    and aggregates these metrics across the dataset for reporting or validation.
    """
    # Load the test dataset
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    num = len(testloader)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    # Initialize metrics storage
    total_dice_coeff1, total_dice_coeff2, total_dice_coeff3, total_dice_coeff4 = 0, 0, 0, 0
    total_dist1, total_dist2, total_dist3, total_dist4 = 0, 0, 0, 0
    
    # Process each batch in the test loader
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # Retrieve image and label from the batch
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
        image_next, image_prev = sampled_batch['next_image'], sampled_batch['prev_image']
        
        dice1, dice2, dice3, dice4, dist1, dist2,dist3, dist4  = test_single_volume(image_next, image, image_prev, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        # Output the metrics for monitoring
        print("dice1 = ",dice1, " dice2 = ", dice2, "dice3= ",dice3,"dice4= ", dice4)
        total_dice_coeff1 = total_dice_coeff1 + dice1
        total_dice_coeff2 = total_dice_coeff2 + dice2
        total_dice_coeff3 = total_dice_coeff3 + dice3
        total_dice_coeff4 = total_dice_coeff4 + dice4
        total_dist1 = total_dist1 + dist1
        total_dist2 = total_dist2 + dist2
        total_dist3 = total_dist3 + dist3
        total_dist4 = total_dist4 + dist4
    
    # Calculate average metrics for all cases
    print(f"dice1={total_dice_coeff1/num}, dice2={total_dice_coeff2/num}, dice3={total_dice_coeff3/num}, dice4={total_dice_coeff4/num}, hd1={total_dist1/num}, hd2={total_dist2/num}, hd3={total_dist3/num}, hd4={total_dist4/num}")
    avg_dice = (total_dice_coeff1 + total_dice_coeff2 + total_dice_coeff3 + total_dice_coeff4) / (4*num)
    print("avg_dice = ",avg_dice)
    # Return the appropriate result based on the validation flag
    if validation:
        return avg_dice
    else:
        return "Testing Finished!"
    
    
    
    
if __name__ == "__main__":
    # Setup GPU/CPU seeds for reproducibility if deterministic mode is enabled
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    # Load dataset configuration based on the provided dataset name
    dataset_config = {
        'CSANet': {
            'Dataset': CSANet_dataset,
            'root_path': '../data/train_npz',
            'volume_path': '../data/',
            'list_dir': './lists',
            'num_classes': 5, 
            'z_spacing': 1,
        },
    }
    
    
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.exp = 'CSANet_'  + str(args.img_size)
    
    
    
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    
    # Initialize and load the ViT model from the specified configuration and saved state
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + args.exp
    
    os.makedirs(log_folder, exist_ok=True)
    # Setup logging and initiate volume inference
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    
    vol_inference(args, net, validation=False)
    
