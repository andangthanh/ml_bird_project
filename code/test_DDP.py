import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

import torchvision
import torch.nn as nn

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
from typing import *
from functools import partial

import matplotlib as mpl

from settings import device
from helpfuncs import camel2snake, listify, accuracy_multi_label, macro_metric_multi_label
from transformations import FreqMask, TimeMask, target_to_one_hot
from databunch import DataBunch, ClassSpecificImageFolder, WholeDataset
from callbacks import StatsCallback, SaveCheckpointCallback, UseCacheCallback
from learner import Runner, Learner, create_BirdNET, create_ResNet50, ddp_create_BirdNET

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):

    mpl.rcParams['figure.dpi']=100
    mpl.use('Agg')

    #====================================
    #====================================
    modelSaveName = "4GPUs"
    architecture = "BirdNET"
    newDir = Path().resolve().parent / f"ddp_testing/{architecture}/{modelSaveName}"
    newDir.mkdir(parents=True, exist_ok=True)
    LEN_CLASSES = 15
    #====================================
    #====================================

    print("1")
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            print("in slurm Zweig")
            args.rank = int(os.environ['SLURM_PROCID'])
            print("args.rank:", args.rank)
            args.gpu = args.rank % torch.cuda.device_count()
            print("gpu:", args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    #################
    # Transformations
    #################

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        #transforms.RandomApply([transforms.RandomResizedCrop(size=(345,128))], p=0.5),
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        transforms.Resize((128,345)), #(64,384)
        #FreqMask(max_mask_size_F=5, num_masks = 5),
        #TimeMask(max_mask_len_T=5, num_masks = 5),
        transforms.Normalize((0.5), (0.5))])

    transform_val = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128,345)), #(64,384)
        transforms.Normalize((0.5), (0.5))])

    #transform = transforms.Compose(
    #    [transforms.Resize((64,384)), #(345,128)
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print("2")

    ################
    # Dataset/Loader
    ################


    path = Path().resolve().parent #workspace folder (/lustre/scratch2/ws/0/s4030475-ml_birds_project/)
    print("3")
    target_t = partial(target_to_one_hot, LEN_CLASSES)
    dropped_classes = ['Actitis macularius', 'Amandava amandava', 'Anthus godlewskii', 
    'Anthus gustavi', 'Branta hutchinsii', 'Bucanetes githagineus', 
    'Calidris fuscicollis', 'Calidris pusilla', 'Caloenas nicobarica', 
    'Charadrius semipalmatus', 'Chrysolophus pictus', 'Decticus verrucivorus', 
    'Dendrocygna bicolor', 'Dendrocygna viduata', 'Gomphocerippus rufus', 
    'Grus japonensis', 'Grus virgo', 'Larus glaucoides', 'Leptophyes calabra', 
    'Melopsittacus undulatus', 'Oenanthe xanthoprymna', 'Omocestus rufipes', 
    'Parnassiana chelmos', 'Parnassiana parnassica', 'Pelecanus occidentalis', 
    'Phaneroptera falcata', 'Pterolepis elymica', 'Rhacocleis maculipedes', 
    'Rhea americana', 'Rhodopechys sanguineus', 'Seiurus aurocapilla', 
    'Sporadiana sporadarum', 'Stenobothrus stigmaticus']

    dropped_classes = []


    small_ds = "lustre/scratch2/ws/0/s4030475-ml_birds_project/bird_data/spectrograms/xeno-canto-data/raw_specs/data"
    europe_split = "lustre/ssd/ws/s4030475-ssd_ml_birds_project/bird_data/europe_split_4000"

    # europe_split: train(857467), val(414050), test(240783)
    # small_ds: train(124399), val(35542) ,test(17772)

    start = time.time()
    train_path = f"/{small_ds}/train"
    #train_ds = WholeDataset(root=train_path, n_samples=124399, n_channels=3, height= 128, width=345 ,dropped_classes=dropped_classes, transform=transform, target_transform=target_t)
    train_ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform, target_transform=target_t)
    #train_dl = DataLoader(dataset=train_ds, batch_size=256, shuffle=True, num_workers=20)

    train_sampler = data.distributed.DistributedSampler(train_ds, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=(train_sampler is None),
            num_workers=10, pin_memory=True, sampler=train_sampler, drop_last=True)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Loading TrainDS:")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    idx2class = {v: k for k, v in train_ds.class_to_idx.items()}
    class_names = train_ds.class_to_idx.keys()
    print("4")

    start = time.time()
    val_path = f"/{small_ds}/val"
    #val_ds = WholeDataset(root=val_path, n_samples=35542, n_channels=3, height= 128, width=345 ,dropped_classes=dropped_classes, transform=transform, target_transform=target_t)
    val_ds = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val, target_transform=target_t)
    #val_dl = DataLoader(dataset=val_ds, batch_size=256, shuffle=False, num_workers=20)

    val_sampler = None
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=(val_sampler is None),
            num_workers=10, pin_memory=True, sampler=val_sampler, drop_last=True)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Loading ValDS:")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    #start = time.time()
    #test_path = f"/{small_ds}/test"
    #test_ds = WholeDataset(root=test_path, n_samples=17772, n_channels=3, height= 128, width=345, dropped_classes=dropped_classes, transform=transform, target_transform=target_t)
    #test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=10)
    #end = time.time()
    #hours, rem = divmod(end-start, 3600)
    #minutes, seconds = divmod(rem, 60)
    #print("Loading TestDS:")
    #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    print("5")
    # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    # https://discuss.pytorch.org/t/help-for-http-error-403-rate-limit-exceeded/125907
    # https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)

    #############
    # Preparation
    #############

    databunch = DataBunch(train_dl, val_dl)
    learn = Learner(*ddp_create_BirdNET(LEN_CLASSES, lr=0.01, args), nn.BCEWithLogitsLoss(), databunch)
    #learn = Learner(*create_ResNet50(LEN_CLASSES, lr=0.01, pretrained=False), nn.BCEWithLogitsLoss(), databunch)

    if args.rank == 0:
        metric_list = [accuracy_multi_label, partial(macro_metric_multi_label,'recall'), partial(macro_metric_multi_label,'precision')]
        stats_cbf = partial(StatsCallback, metric_list, newDir)
        checkpoint_cbf = partial(SaveCheckpointCallback, newDir)
        run = Runner(cb_funcs=[stats_cbf, checkpoint_cbf], rank = args.rank)
    else:
        run = Runner(rank = args.rank)


    ### model ###
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            #torch.cuda.set_device(args.gpu)
            #model.cuda(args.gpu)
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            #model_without_ddp = model.module
            print("in if")
        else:
            #model.cuda()
            #model = torch.nn.parallel.DistributedDataParallel(model)
            #model_without_ddp = model.module
            print("in else")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    torch.backends.cudnn.benchmark = True

    ##########
    # Training
    ##########

    start = time.time()
    print("6")
    print("Begin training.")

    run.fit(15, learn)

    print("end of training")

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time:")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == '__main__':
    args = parse_args()
    main(args)