import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

import math
from pathlib import Path
from typing import *
import shutil
from functools import partial
import collections
# from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import random

# from IPython import display

import os
import pickle
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
mpl.use('Agg')

#====================================
#====================================
modelSaveName = "SpecAug20"
architecture = "BirdNET"
newDir = Path().resolve().parent / f"figures_and_models/{architecture}/{modelSaveName}"
newDir.mkdir(parents=True, exist_ok=True)
#====================================
#====================================

FILTERS = [8, 16, 32, 64, 128]
#KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
KERNEL_SIZES = [5, 3, 3, 3, 3]
RESNET_K = 4
RESNET_N = 3
LEN_CLASSES = 15

print("1")
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
print("2")

#export
import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def accuracy_multi_label(pred, yb): return (torch.argmax(pred, dim=1)==torch.argmax(yb, dim=1)).float().mean()

class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
        
    @property
    def train_ds(self): 
        return self.train_dl.dataset
    
    @property
    def valid_ds(self):
        return self.valid.dl.dataset


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class ResNet50(nn.Module):
    
    def __init__(self, n_classes, pretrained=true):
        super(ResNet50, self).__init__()
        if pretrained:
            resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = torchvision.models.resnet50(weights=None)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=resnet.fc.in_features, out_features=nn_classes))
        self.base_model = resnet

    def forward(self, x):
        return self.base_model(x)


class Resblock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):
        super(Resblock, self).__init__()

        self.block_id = block_id
        self.preactivated = preactivated
        self.stride = stride
        self.block_id = block_id
        self.name = name

        # Bottleneck Convolution
        self.conv1 = Conv2dSame(in_channels, out_channels=in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # First Convolution   
        self.conv2 = Conv2dSame(in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=(stride, stride), stride=(stride, stride))

        # Dropout Layer
        self.drop = nn.Dropout()

        # Second Convolution
        self.conv3 = Conv2dSame(in_channels, filters, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(filters)

        # Average Pooling
        self.pool2 = nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride), count_include_pad=False)
        
        # Shortcut Convolution
        self.conv4 = Conv2dSame(in_channels, filters, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(filters)

        self.relu = nn.ReLU()
        # Pre-activation

    
    def forward(self, x):

        if self.block_id > 1:
            pre = self.relu(x)
        else:
            pre = x 

        # Pre-activated shortcut?
        if self.preactivated:
            x = pre

        # Bottleneck Convolution
        if self.stride > 1:
            pre = self.conv1(pre)
            pre = self.bn1(pre)
            pre = self.relu(pre)
        
        # First Convolution
        net = self.conv2(pre)
        net = self.bn2(net) 
        net = self.relu(net)

        # Pooling layer
        if self.stride > 1:
            net = self.pool1(net)

        # Dropout Layer
        net = self.drop(net)     

        # Second Convolution
        net = self.conv3(net)
        net = self.bn3(net)
        net = self.relu(net)

        # Shortcut Layer
        if not list(net.size()) == list(x.size()):

            # Average pooling
            shortcut = self.pool2(x)

            # Shortcut convolution
            shortcut = self.conv4(shortcut)
            shortcut = self.bn4(shortcut)   
            
        else:

            # Shortcut = input
            shortcut = x
        
        # Merge Layer
        out = net + shortcut

        return out


class BirdNET(nn.Module):
    
    def __init__(self, n_classes):
        super(BirdNET, self).__init__()

        # Pre-processing stage
        self.conv1 = Conv2dSame(in_channels=1, out_channels=FILTERS[0] * RESNET_K, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(int(FILTERS[0] * RESNET_K))

        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Residual Stacks
        self.resblocks = []
        for i in range(1, len(FILTERS)):
            self.resblocks.append(Resblock(in_channels=int(FILTERS[i-1] * RESNET_K),
                                           filters=int(FILTERS[i] * RESNET_K),
                                           kernel_size=KERNEL_SIZES[i],
                                           stride=2,
                                           preactivated=True,
                                           block_id=i,
                                           name='BLOCK ' + str(i) + '-1').cuda())

            for j in range(1, RESNET_N):
                self.resblocks.append(Resblock(in_channels=int(FILTERS[i] * RESNET_K),
                                               filters=int(FILTERS[i] * RESNET_K),
                                               kernel_size=KERNEL_SIZES[i],
                                               preactivated=False,
                                               block_id=i+j,
                                               name='BLOCK ' + str(i) + '-' + str(j + 1)).cuda())
        
        self.bn2 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K))

        # Classification Branch     
        self.conv2 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K), out_channels=int(FILTERS[-1] * RESNET_K), kernel_size=(21,2))
        self.bn3 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K))
        self.drop1 = nn.Dropout()
        # Dense Convolution  
        self.conv3 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K), out_channels=int(FILTERS[-1] * RESNET_K * 2), kernel_size=1)
        self.bn4 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K * 2))
        self.drop2 = nn.Dropout()
        # Class Convolution
        self.conv4 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K * 2), out_channels=n_classes, kernel_size=1)

        # Pooling  (kernelsize == last feature map size, depends on size of images)
        #self.pool2 = nn.AvgPool2d((1,3))

        # Flatten the output
        #self.flatten = nn.Flatten()

        # Sigmoid Output
        # self.sig = nn.Sigmoid()

        self.relu = nn.ReLU()
                                               
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.resblocks:
            x = block(x)

        x = self.bn2(x)
        x = self.relu(x)

        # Classification Branch
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.conv4(x)

        x = torch.logsumexp(x, (2,3))

        # Global Pooling
        #x = self.pool2(x)

        # Flatten Output
        #x = self.flatten(x)

        # Sigmoid Output
        # x = self.sig(x)

        return x


def create_BirdNET(n_classes, lr=0.01):
    """Creates model, optimizer"""

    model = BirdNET(n_classes)
    model.to(device)
    model.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


def create_ResNet50(n_classes, lr=0.01, pretrained=true):
    """Creates model, optimizer"""

    model = ResNet50(n_classes, pretrained)
    model.to(device)
    model.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


class Learner:
    """Class for Storing all necessary objects"""
    def __init__(self, model, optimizer, criterion, databunch):
        self.model = model
        self.databunch = databunch
        self.criterion = criterion
        self.optimizer = optimizer


class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    
    # delegate object to run object if cant find attribute in Callback class
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0
    
    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters  #epoch as float 
        self.run.n_iter   += 1
        
    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


class Stats():
    def __init__(self, metrics, in_train):
        self.metrics = listify(metrics)
        self.in_train = in_train
        self.hist_metrics = collections.defaultdict(list)

    @property
    def all_stats(self):  return [self.tot_loss] + self.tot_mets
    @property
    def avg_stats(self):  return [o/self.count for o in self.all_stats]
        
    def reset(self):
        self.tot_loss = 0.
        self.count = 0
        self.tot_mets = [0.] * len(self.metrics)
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumulate(self, run):
        bn = run.xb.shape[0]  # batch size
        self.tot_loss += run.loss.item() * bn  # adding batch loss
        self.count += bn  # adding all batchsizes for dividing in avg_stats attribute
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class StatsCallback(Callback):
    _order = 1
    def __init__(self, metrics):
        self.train_stats = Stats(metrics, True)
        self.valid_stats = Stats(metrics, False)
        self.plotCount = 1

    def begin_fit(self):
        if run.resume:
            self.plotCount = run.epoch + 1
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        stats.accumulate(self.run)
            
    def after_epoch(self):
        for i in range(len(self.train_stats.avg_stats)):
            self.train_stats.hist_metrics[i].append(self.train_stats.avg_stats[i])
            self.valid_stats.hist_metrics[i].append(self.valid_stats.avg_stats[i])

        if run.epoch > 0 and run.epoch%1==0:
            #=====================================================================================================================================#
            #=====================================================================================================================================#
            #=====================================================================================================================================#
            # Check newDir at the top before training to prevent overwriting existing evaluation data!
            save_path = newDir / "plots"
            #=====================================================================================================================================#
            #=====================================================================================================================================#
            #=====================================================================================================================================#
            
            # Clear the previous plot
            # clear_output(wait=True)
            N = np.arange(0, len(self.train_stats.hist_metrics[0]))
            
            # You can chose the style of your preference
            plt.style.use("seaborn")
            
            # Plot train loss, val loss against epochs passed
            cut_at = 20
            plt.figure(figsize=(6,4))
            plt.title("Loss over epoch No. {}".format(run.epoch))
            t=np.stack(self.train_stats.hist_metrics[0])
            t=np.ma.masked_where(t > cut_at, t)
            v=np.stack(self.valid_stats.hist_metrics[0])
            v=np.ma.masked_where(v > cut_at, v)
            plt.plot(N, t, label = "Training Loss", c='cornflowerblue')
            plt.plot(N, v, label = "Valid Loss", c='orange')
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")
            plt.tight_layout()
            (save_path / "loss").mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"loss/loss{self.plotCount}.eps", format = 'eps')
            plt.savefig(save_path / f"loss/loss{self.plotCount}.pdf", format = 'pdf')
            plt.savefig(save_path / f"loss/loss{self.plotCount}.png", format = 'png')
            plt.close()

            # Plot train acc, val acc
            plt.figure(figsize=(6,4))
            plt.title("Accuracy over epoch No. {}".format(run.epoch))
            plt.plot(N, self.train_stats.hist_metrics[1], label = "Training Accuracy", c='cornflowerblue')
            plt.plot(N, self.valid_stats.hist_metrics[1], label = "Valid Accuracy", c='orange')
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.ylim([0.0, 1.05])
            plt.legend(loc="upper left")
            (save_path / "acc").mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"acc/acc{self.plotCount}.eps", format = 'eps')
            plt.savefig(save_path / f"acc/acc{self.plotCount}.pdf", format = 'pdf')
            plt.savefig(save_path / f"acc/acc{self.plotCount}.png", format = 'png')
            plt.close()
            
            self.plotCount += 1
            
            #plt.show()


class SaveCheckpointCallback(Callback):
    _order = 2
    def __init__(self):
        self.best_loss = 10000000.0
    
    def after_epoch(self):
        is_best = run.stats.valid_stats.avg_stats[0] < self.best_loss
        self.best_loss = min(run.stats.valid_stats.avg_stats[0], self.best_loss)



        # Eventuell nur speichern, wenn bester Loss, jedes mal Checkpoint speichern könnte
        # lange dauern  

        self.save_checkpoint({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_loss_his' : run.stats.train_stats.hist_metrics[0],
            'valid_loss_his' : run.stats.valid_stats.hist_metrics[0],
            'train_acc' : run.stats.train_stats.hist_metrics[1],
            'valid_acc' : run.stats.valid_stats.hist_metrics[1],
        }, is_best, filename=newDir / "checkpoint.pth.tar")
        
    def after_fit(self):

        torch.save({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_loss_his' : run.stats.train_stats.hist_metrics[0],
            'valid_loss_his' : run.stats.valid_stats.hist_metrics[0],
            'train_acc' : run.stats.train_stats.hist_metrics[1],
            'valid_acc' : run.stats.valid_stats.hist_metrics[1],
        }, newDir / "final_epoch_model.pth.tar")
            
        
    def save_checkpoint(self, model_state, is_best, filename="checkpoint.pth.tar"):
        torch.save(model_state, filename)
        if is_best:
            shutil.copyfile(filename, newDir / "model_best.pth.tar")   


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            
            # sets attribute for every Callback
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    # delegating attributes to learn object
    @property
    def model(self):           return self.learn.model
    @property
    def databunch(self):       return self.learn.databunch
    @property
    def criterion(self):       return self.learn.criterion
    @property
    def optimizer(self):       return self.learn.optimizer

    def one_batch(self, batch_data):
        self.xb,self.yb = batch_data[0].to(device), batch_data[1].to(device)
        if self('begin_batch'): return
        self.optimizer.zero_grad()
        self.pred = self.model(self.xb)
        if self('after_pred'): return
        self.loss = self.criterion(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.optimizer.step()
        if self('after_step'): return

    def all_batches(self, dl):
        self.iters = len(dl) #number of batches
        for i, batch_data in enumerate(dl, 0):
            if self.stop: break
            self.one_batch(batch_data)
            self('after_batch')
        self.stop=False

    def fit(self, epochs, learn, resume=False, start_epoch=0):
        self.epochs,self.learn,self.resume = epochs,learn,resume

        try:
            for cb in self.cbs: cb.set_runner(self)
            if resume:
                self.epoch=start_epoch
            else:
                self.epoch = 0
            if self('begin_fit'): return
            while self.epoch < epochs:
                if not self('begin_epoch'): self.all_batches(self.databunch.train_dl)

                # validation mode
                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.databunch.valid_dl)
                self.epoch += 1
                if self('after_epoch'): break
                 
        finally:
            self('after_fit')
            #self.learn = None

    # used for Callbacks. Gets executed for example when self('after_epoch'). Callbacks get executed sorted in _order
    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False


def _time_mask(mel_spectro, max_mask_len_T=27, num_masks=1, replace_with_zero=False):
    """masked a maximum of 'max_mask_size' consecutive time-steps(width)"""

    len_mel_spectro = mel_spectro.shape[2]
    
    channels = mel_spectro.shape[0]
    
    for i in range(0, num_masks):
        
        # 't_mask_len' randomly chosen from uniform distribution from 0 to 'max_mask_len_T'
        t_mask_len = random.randrange(0, max_mask_len_T)
        
        # avoids randrange error if mask length is bigger than length of mel-spectrogram, return original mel-spectrogram
        if (t_mask_len >= len_mel_spectro): 
            return mel_spectro
        
        # begin of mask
        t_mask_begin = random.randrange(0, len_mel_spectro - t_mask_len)

        # avoids randrange error if values are equal and range is empty
        if (t_mask_begin == t_mask_begin + t_mask_len): 
            return mel_spectro
        
        # end of mask
        t_mask_end = random.randrange(t_mask_begin, t_mask_begin + t_mask_len)
        
        if (replace_with_zero):
            for c in range(channels):
                mel_spectro[c][:,t_mask_begin:t_mask_end] = 0
        else:
            for c in range(channels):
                mel_spectro[c][:,t_mask_begin:t_mask_end] = mel_spectro[c].mean()
                
    return mel_spectro


class FreqMask(nn.Module):
    def  __init__(self, max_mask_size_F=20, num_masks=1, replace_with_zero=False):
        super().__init__()
        self.max_mask_size_F = max_mask_size_F
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def forward(self, mel_spectro):
        """masked a maximum of 'max_mask_size' consecutive mel frequency channels(height)"""
        
        # number of mel channels
        num_mels = mel_spectro.shape[1]
        channels = mel_spectro.shape[0]
        
        for i in range(0, self.num_masks):  
            
            # 'f_mask_size' randomly chosen from uniform distribution from 0 to 'max_mask_size_F'
            f_mask_size = random.randrange(0,  self.max_mask_size_F)
                
            # avoids randrange error if mask size is bigger than number of mel channels, return original mel-spectrogram
            if (f_mask_size >= num_mels): 
                return mel_spectro
            
            # begin of mask
            f_mask_begin = random.randrange(0, num_mels - f_mask_size)

            # avoids randrange error if values are equal and mask size is 0, return original mel-spectrogram
            if (f_mask_begin == f_mask_begin + f_mask_size) : 
                return mel_spectro

            # begin of mask
            f_mask_end = random.randrange(f_mask_begin, f_mask_begin + f_mask_size) 
            
            if (self.replace_with_zero):
                for c in range(channels):
                    mel_spectro[c][f_mask_begin:f_mask_end] = 0
            else: 
                for c in range(channels):
                    mel_spectro[c][f_mask_begin:f_mask_end] = mel_spectro[c].mean()
            
        return mel_spectro


class TimeMask(nn.Module):
    def  __init__(self, max_mask_len_T=20, num_masks=1, replace_with_zero=False):
        super().__init__()
        self.max_mask_len_T = max_mask_len_T
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def forward(self, mel_spectro):
        """masked a maximum of 'max_mask_size' consecutive time-steps(width)"""

        len_mel_spectro = mel_spectro.shape[2]
        channels = mel_spectro.shape[0]
        
        for i in range(0, self.num_masks):
            
            # 't_mask_len' randomly chosen from uniform distribution from 0 to 'max_mask_len_T'
            t_mask_len = random.randrange(0, self.max_mask_len_T)
            
            # avoids randrange error if mask length is bigger than length of mel-spectrogram, return original mel-spectrogram
            if (t_mask_len >= len_mel_spectro): 
                return mel_spectro
            
            # begin of mask
            t_mask_begin = random.randrange(0, len_mel_spectro - t_mask_len)

            # avoids randrange error if values are equal and range is empty
            if (t_mask_begin == t_mask_begin + t_mask_len): 
                return mel_spectro
            
            # end of mask
            t_mask_end = random.randrange(t_mask_begin, t_mask_begin + t_mask_len)
            
            if (self.replace_with_zero):
                for c in range(channels):
                    mel_spectro[c][:,t_mask_begin:t_mask_end] = 0
            else:
                for c in range(channels):
                    mel_spectro[c][:,t_mask_begin:t_mask_end] = mel_spectro[c].mean()
                    
        return mel_spectro


transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
    #transforms.RandomApply([transforms.RandomResizedCrop(size=(345,128))], p=0.5),
    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
    transforms.Resize((345,128)), #(64,384)
    transforms.ToTensor(),
    FreqMask(num_masks = 2),
    TimeMask(num_masks = 2),
    transforms.Normalize((0.5), (0.5))])

transform_val = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
    transforms.Resize((345,128)), #(64,384)
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))])

#transform = transforms.Compose(
#    [transforms.Resize((64,384)), #(345,128)
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def target_to_one_hot(target):
    NUM_CLASS = 15
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot

import time
start = time.time()

path = Path().resolve().parent #workspace folder (/lustre/scratch2/ws/0/s4030475-ml_birds_project/)
print("3")
train_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/train"
train_ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform, target_transform=target_to_one_hot)
train_dl = DataLoader(dataset=train_ds, batch_size=256, shuffle=True, num_workers=10)

idx2class = {v: k for k, v in train_ds.class_to_idx.items()}
print("4")
val_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/val"
val_ds = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val, target_transform=target_to_one_hot)
val_dl = DataLoader(dataset=val_ds, batch_size=128, shuffle=False, num_workers=10)

test_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/test"
test_ds = torchvision.datasets.ImageFolder(root=test_path, transform=transform_val, target_transform=target_to_one_hot)
test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=10)

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



print("5")
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# https://discuss.pytorch.org/t/help-for-http-error-403-rate-limit-exceeded/125907
# https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511

# model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)



databunch = DataBunch(train_dl, val_dl)
#learn = Learner(*create_BirdNET(LEN_CLASSES, lr=0.01), nn.BCEWithLogitsLoss(), databunch)
learn = Learner(*create_ResNet50(LEN_CLASSES, lr=0.01), nn.BCEWithLogitsLoss(), databunch)


stats_cbf = partial(StatsCallback, accuracy_multi_label)
checkpoint_cbf = partial(SaveCheckpointCallback)

run = Runner(cb_funcs=[stats_cbf, checkpoint_cbf])




# model = BirdNET()
# model.to(device)
# model.cuda()
# # criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.005)



# def multi_acc(y_pred, y_test):  # mit CrossEntropyLoss()
#     y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
#     _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
#     correct_pred = (y_pred_tags == y_test).float()
#     acc = correct_pred.sum() / len(correct_pred)
#     acc = torch.round(acc * 100)
#     return acc


# def multi_acc_multi_label(y_pred, y_test): # mit BCEWithLogitsLoss()
#     y_pred_sig = torch.sigmoid(y_pred)
#     y_pred_tags = torch.argmax(y_pred_sig, dim = 1)
#     y_test_tags = torch.argmax(y_test, dim = 1)
#     correct_pred = (y_pred_tags == y_test_tags).float()
#     acc = correct_pred.sum() / len(correct_pred)
#     #acc = torch.round(acc * 100)
#     return acc

# accuracy_stats = {
#     "train": [],
#     "val": []
# }
# loss_stats = {
#     "train": [],
#     "val": []
# }





start = time.time()
print("6")
print("Begin training.")

run.fit(300, learn)

# epoch = 3

# for epoch in range(epoch):
#     print("epoch: ", epoch)
#     running_epoch_loss = 0 
#     running_epoch_acc = 0
#     total_bs = 0 # accumulated batchsizes (should equal to number of samples)
#     model.train()

#     for i, batch_data in enumerate(train_dl, 0):
#         xb, yb = batch_data[0].to(device), batch_data[1].to(device)
#         optimizer.zero_grad()
#         pred = model(xb)

#         train_loss = criterion(pred, yb)
#         train_acc = multi_acc_multi_label(pred, yb)

#         train_loss.backward()
#         optimizer.step()

#         bs = xb.shape[0] # batchsize
#         total_bs += bs
        
#         running_epoch_loss += train_loss.item() * bs
#         running_epoch_acc += train_acc.item() * bs

#     with torch.no_grad():

#         val_epoch_loss = 0
#         val_epoch_acc = 0
#         val_total_bs = 0
#         model.eval()
        
#         for i, batch_data in enumerate(val_dl,0):
#             xb, yb = batch_data[0].to(device), batch_data[1].to(device)

#             pred = model(xb)
#             val_loss = criterion(pred, yb)
#             val_acc = multi_acc_multi_label(pred, yb)

#             bs = xb.shape[0] # batchsize
#             val_total_bs += bs

#             val_epoch_loss += val_loss.item() *bs
#             val_epoch_acc += val_acc.item() * bs

#     loss_stats["train"].append(running_epoch_loss/total_bs) # average loss per batch
#     loss_stats["val"].append(val_epoch_loss/val_total_bs)

#     accuracy_stats["train"].append(running_epoch_acc/total_bs)
#     accuracy_stats["val"].append(val_epoch_acc/val_total_bs)

print("end of training")

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))






# train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

# save_path = path / "figures"
# # Plot line charts
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
# sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
# sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
# fig.savefig(save_path / "loss_accTEST.eps", format = 'eps')
# fig.savefig(save_path / "loss_accTEST.pdf", format = 'pdf')
# fig.savefig(save_path / "loss_accTEST.png", format = 'png')

# print("end of loss graph")

y_pred_list = []
y_true_list = []
with torch.no_grad():
    for i, batch_data in enumerate(test_dl,0):
        xb, yb = batch_data[0].to(device), batch_data[1].to(device)
        y_test_pred = run.learn.model(xb)

        y_pred_sig = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.argmax(y_test_pred, dim = 1)

        y_test_tag = torch.argmax(yb, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_test_tag.cpu().numpy())


torch.save({
    'test_pred_list': y_pred_list,
    'test_true_list': y_true_list
}, newDir / "end_test_inference.pth.tar")


print(classification_report(y_true_list, y_pred_list))
print(confusion_matrix(y_true_list, y_pred_list))

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(14,10))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax, fmt="d")
(newDir / "plots/cm").mkdir(parents=True, exist_ok=True)
fig.savefig(newDir / "plots/cm/cmTEST.eps", format = 'eps')
fig.savefig(newDir / "plots/cm/cmTEST.pdf", format = 'pdf')
fig.savefig(newDir / "plots/cm/cmTEST.png", format = 'png')
plt.close()
