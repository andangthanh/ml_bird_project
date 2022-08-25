import torch
import math
from typing import *
from sklearn.metrics import classification_report
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

def macro_metric_multi_label(metric, pred, yb):
    pred_list = [x.numpy() for x in pred]
    true_list = [x.numpy() for x in yb]
    return classification_report(true_list, pred_list, output_dict=True)['macro avg'][metric]




