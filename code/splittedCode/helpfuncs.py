import torch
import math
from typing import *
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import re
import numpy as np

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

def macro_metric_multi_label(metric, pred, yb):
    pred = torch.sigmoid(pred)
    pred_list = [x.detach().cpu().numpy() for x in pred]
    pred_list = np.around(pred_list)
    true_list = [x.detach().cpu().numpy() for x in yb]
    return classification_report(true_list, pred_list, output_dict=True)['macro avg'][metric]

def f1_score_multi_label(pred, yb):
    pred = torch.sigmoid(pred)
    pred_list = [x.detach().cpu().numpy() for x in pred]
    pred_list = np.around(pred_list)
    true_list = [x.detach().cpu().numpy() for x in yb]
    return f1_score(true_list, pred_list, average='weighted', zero_division=1)

def recall_score_multi_label(pred, yb):
    pred = torch.sigmoid(pred)
    pred_list = [x.detach().cpu().numpy() for x in pred]
    pred_list = np.around(pred_list)
    true_list = [x.detach().cpu().numpy() for x in yb]
    return recall_score(true_list, pred_list, average='weighted', zero_division=1)

def precision_score_multi_label(pred, yb):
    pred = torch.sigmoid(pred)
    pred_list = [x.detach().cpu().numpy() for x in pred]
    pred_list = np.around(pred_list)
    true_list = [x.detach().cpu().numpy() for x in yb]
    return precision_score(true_list, pred_list, average='weighted', zero_division=1)

def accuracy_score_multi_label(pred, yb):
    pred = torch.sigmoid(pred)
    pred_list = [x.detach().cpu().numpy() for x in pred]
    pred_list = np.around(pred_list)
    true_list = [x.detach().cpu().numpy() for x in yb]
    return accuracy_score(true_list, pred_list)

def f1_categorical(pred, yb):
    pred_list = torch.argmax(pred,dim=1).detach().cpu().numpy()
    true_list = yb.detach().cpu().numpy()
    return f1_score(true_list, pred_list, average='micro', zero_division=1)

def recall_categorical(pred, yb):
    pred_list = torch.argmax(pred,dim=1).detach().cpu().numpy()
    true_list = yb.detach().cpu().numpy()
    return recall_score(true_list, pred_list, average='weighted', zero_division=1)

def precision_categorical(pred, yb):
    pred_list = torch.argmax(pred,dim=1).detach().cpu().numpy()
    true_list = yb.detach().cpu().numpy()
    return precision_score(true_list, pred_list, average='weighted', zero_division=1)

def accuracy_categorical(pred, yb):
    pred_list = torch.argmax(pred,dim=1).detach().cpu().numpy()
    true_list = yb.detach().cpu().numpy()
    return f1_score(true_list, pred_list)