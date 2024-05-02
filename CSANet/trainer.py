import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss_function import DiceLoss
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import test_single_volume
from visualization import save_visualization
import cv2
import torch.backends.cudnn as cudnn
from datasets.dataset_CSANet import CSANet_dataset, RandomGenerator
from test import vol_inference 






def trainer_CSANet(args, model, snapshot_path):
    
    """
    Trains the CSANet model with the specified parameters and dataset, performing evaluations and saving the model state based on performance metrics.

    Parameters:
    - args (Namespace): Configuration containing all settings for the training process, such as dataset paths, learning rates, batch sizes, and more.
    - model (torch.nn.Module): The neural network model to be trained.
    - snapshot_path (str): Directory path where training snapshots (model states and logs) will be saved.

    Returns:
    - str: A message indicating that training has finished.

    The function initializes training setup, logs configurations, and enters a training loop where it continually feeds data through the model, computes losses, updates the model's weights, and logs the results. It also evaluates the model periodically using the `vol_inference` function and saves the model state when performance improves. The summary of training progress is saved using TensorBoard.
    """
    
    # Configure logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Set training parameters from args
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Initialize dataset and dataloader
    db_train = CSANet_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train_image",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        # Seed each worker for reproducibility
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # Use DataParallel for multi-GPU training
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # Define loss functions and optimizer
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    folder_path = "./training_result"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    #vol_inference(args, model, validation=False)
    # Training loop
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_next, image_prev = sampled_batch['next_image'], sampled_batch['prev_image']
            
            # Ensure all tensors are on the same device
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            image_next_batch, image_prev_batch = image_next.cuda(), image_prev.cuda()
            
            # Forward pass
            outputs = model(image_prev_batch, image_batch, image_next_batch)
            
            # Calculate loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Visualization and logging
            save_visualization(outputs, label_batch, epoch_num, i_batch)
            
            # Learning rate adjustment
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        # End-of-epoch validation and checkpointing
        if epoch_num > 10 and (epoch_num % 5 == 0 or epoch_num == 39):
            avg_dice = vol_inference(args, model, validation=True)
            if avg_dice > best_performance:
                best_performance = avg_dice
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"Saved new best model to {save_mode_path}")
            
    vol_inference(args, model, validation=False)
    writer.close()
    return "Training Finished!"
