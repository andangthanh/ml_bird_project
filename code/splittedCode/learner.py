import torch.optim as optim
from models import BirdNET, ResNet50
from settings import device
from helpfuncs import *
from callbacks import TrainEvalCallback
import torch
import numpy as np
import random
from datetime import datetime
import torch.distributed as dist

def create_BirdNET(n_classes, lr=0.01, sgd=False, momentum=0.9, weight_decay=0):
    """Creates model, optimizer"""

    model = BirdNET(n_classes)
    model.to(device)
    model.cuda()
    
    if sgd == True:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


def ddp_create_BirdNET(n_classes, lr=0.01, args=None, sgd=False, momentum=0.9, weight_decay=0):
    """Creates model, optimizer"""

    model = BirdNET(n_classes)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #model_without_ddp = model.module

    if sgd == True:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


def create_ResNet50(n_classes, lr=0.01, pretrained=True, sgd=False, momentum=0.9, weight_decay=0):
    """Creates model, optimizer"""

    model = ResNet50(n_classes, pretrained)
    model.to(device)
    model.cuda()
    
    if sgd == True:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


def ddp_create_ResNet50(n_classes, lr=0.01, pretrained=True, args=None, sgd=False, momentum=0.9, weight_decay=0):
    """Creates model, optimizer"""

    model = ResNet50(n_classes, pretrained)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #model_without_ddp = model.module

    if sgd == True:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


class Learner:
    """Class for Storing all necessary objects"""
    def __init__(self, model, optimizer, criterion, databunch):
        self.model = model
        self.databunch = databunch
        self.criterion = criterion
        self.optimizer = optimizer


class Runner():
    def __init__(self, cbs=None, cb_funcs=None, rank=None, distributed=None):
        self.rank = rank
        self.distributed = distributed
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
        if not self.in_train and self.distributed:
            self.pred = self.model.module(self.xb)
        else:
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
                if self.distributed != None:
                    #print("Rank: ", self.rank, " beginn epoch   ", self.epoch, " at: ",datetime.now().strftime("%H:%M:%S"))
                    np.random.seed(self.epoch)
                    random.seed(self.epoch)
                    self.databunch.train_dl.sampler.set_epoch(self.epoch)
                if not self('begin_epoch'): self.all_batches(self.databunch.train_dl)

                # validation mode
                #print("Rank: ", self.rank, " finished epoch ", self.epoch, " at: ",datetime.now().strftime("%H:%M:%S"))
                if self.rank == 0 or self.rank == None:
                    with torch.no_grad():
                        if not self('begin_validate'): self.all_batches(self.databunch.valid_dl)
                        #print("     Rank: ", self.rank, " epoch: ", self.epoch)
                        if self.distributed:
                            dist.barrier()
                else: 
                    #print("     Rank: ", self.rank, " epoch: ", self.epoch)
                    if self.distributed:
                        dist.barrier()
                self.epoch += 1
                if self('after_epoch'): break
                 
        finally:
            self('after_fit')
            #self.learn = None

    # used for Callbacks. Gets executed for example when self('after_epoch'). Callbacks get executed sorted in _order
    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True # if a callback function returns True this statement returns True
        return False # if all callback functions return nothing or False the __call__ function will return False

