import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
from typing import *
from functools import partial

import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
mpl.use('Agg')

from settings import device
from helpfuncs import camel2snake, listify, accuracy_multi_label
from transformations import FreqMask, TimeMask, target_to_one_hot
from databunch import DataBunch
from callbacks import StatsCallback, SaveCheckpointCallback
from learner import Runner, Learner, create_BirdNET, create_ResNet50


#====================================
#====================================
modelSaveName = "SpecAug5Mask"
architecture = "ResNet50"
newDir = Path().resolve().parent / f"figures_and_models/{architecture}/{modelSaveName}"
newDir.mkdir(parents=True, exist_ok=True)
LEN_CLASSES = 15
#====================================
#====================================

print("1")
np.random.seed(0)
torch.manual_seed(0)


#################
# Transformations
#################

transform = transforms.Compose(
    [#transforms.Grayscale(num_output_channels=1),
    #transforms.RandomApply([transforms.RandomResizedCrop(size=(345,128))], p=0.5),
    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
    transforms.Resize((345,128)), #(64,384)
    transforms.ToTensor(),
    FreqMask(max_mask_size_F=5, num_masks = 5),
    TimeMask(max_mask_len_T=5, num_masks = 5),
    transforms.Normalize((0.5), (0.5))])

transform_val = transforms.Compose(
    [#transforms.Grayscale(num_output_channels=1),
    transforms.Resize((345,128)), #(64,384)
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))])

#transform = transforms.Compose(
#    [transforms.Resize((64,384)), #(345,128)
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("2")
################
# Dataset/Loader
################

import time
start = time.time()

path = Path().resolve().parent #workspace folder (/lustre/scratch2/ws/0/s4030475-ml_birds_project/)
print("3")
target_t = partial(target_to_one_hot, LEN_CLASSES)
train_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/train"
train_ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform, target_transform=target_t)
train_dl = DataLoader(dataset=train_ds, batch_size=256, shuffle=True, num_workers=10)

idx2class = {v: k for k, v in train_ds.class_to_idx.items()}
print("4")
val_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/val"
val_ds = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val, target_transform=target_t)
val_dl = DataLoader(dataset=val_ds, batch_size=128, shuffle=False, num_workers=10)

test_path = path / "bird_data/spectrograms/xeno-canto-data/raw_specs/data/test"
test_ds = torchvision.datasets.ImageFolder(root=test_path, transform=transform_val, target_transform=target_t)
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


#############
# Preparation
#############

databunch = DataBunch(train_dl, val_dl)
#learn = Learner(*create_BirdNET(LEN_CLASSES, lr=0.01), nn.BCEWithLogitsLoss(), databunch)
learn = Learner(*create_ResNet50(LEN_CLASSES, lr=0.01), nn.BCEWithLogitsLoss(), databunch)

stats_cbf = partial(StatsCallback, accuracy_multi_label, newDir)
checkpoint_cbf = partial(SaveCheckpointCallback, newDir)
run = Runner(cb_funcs=[stats_cbf, checkpoint_cbf])


##########
# Training
##########

start = time.time()
print("6")
print("Begin training.")

run.fit(300, learn)

print("end of training")

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



##############
# Interference
##############

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


#################
# ConfusionMatrix
#################

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
