# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

def main():

    args = cfg.parse_args()

    seed = args.seed
    set_seed(seed)

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(args.epochs):

        if epoch < 5:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == args.epochs-1:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if edice > best_dice:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                is_best = False

    writer.close()


if __name__ == '__main__':
    main()
