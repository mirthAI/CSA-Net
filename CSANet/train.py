import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_CSANet
from datasets.dataset_CSANet import CSANet_dataset



"""
This script initializes and runs training for the CSANet segmentation model using Vision Transformer (ViT) architecture.
It configures the training environment, sets up the data loading for a medical imaging dataset, and initializes the model
with predefined or specified hyperparameters. The script is designed to be run with command-line arguments that allow 
customization of various parameters including data paths, model specifics, and training settings.

Command-Line Arguments:
- root_path: Directory containing training data.
- dataset: Identifier for the dataset used, affecting certain preset configurations.
- list_dir: Directory containing lists of training data specifics.
- num_classes: Number of classes for segmentation.
- volume_path: Path to validation data for model evaluation.
- max_iterations: Total number of iterations the training should run.
- max_epochs: Maximum number of epochs for which the model trains.
- batch_size: Number of samples in each batch.
- n_gpu: Number of GPUs available for training.
- deterministic: Flag to ensure deterministic results, useful for reproducibility.
- base_lr: Base learning rate for the optimizer.
- img_size: Dimensions of the input images for the model.
- seed: Random seed for initialization to ensure reproducibility.
- n_skip: Number of skip connections in the ViT model.
- vit_name: Name of the Vision Transformer configuration to be used.
- vit_patches_size: Size of patches used in the ViT model.

The script supports customization of the training process through these parameters and uses a pre-defined configuration
for setting up the model, dataset, and training operations based on the provided dataset name.
"""

# Setup command-line interface
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='CSANet', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network') # class change -----------------
parser.add_argument('--volume_path', type=str,
                    default='../data', help='root dir for validation volume data')
parser.add_argument('--max_iterations', type=int,
                    default=300000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()


if __name__ == "__main__":
    # Configure deterministic behavior for reproducibility if specified
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
    dataset_config = {
        'CSANet': {
            'Dataset': CSANet_dataset,
            'root_path': '../data/train_npz',
            'volume_path': '../data',
            'list_dir': './lists',
            'num_classes': 5, 
            'z_spacing': 1,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.exp = 'CSANet_'+ str(args.img_size)
    
    # Build snapshot path based on the configuration and command-line arguments
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path

    print("snapshot path = ", snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    # Load Vision Transformer with the specific configuration
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # Load initial weights if pretrained path is provided
    net.load_from(weights=np.load(config_vit.pretrained_path))
    # Start training using the specified trainer for the dataset
    trainer = {'CSANet': trainer_CSANet}
    trainer[dataset_name](args, net, snapshot_path)
