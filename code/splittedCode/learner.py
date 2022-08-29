import torch.optim as optim
from models import BirdNET, ResNet50
from settings import device
from helpfuncs import *
from callbacks import TrainEvalCallback

def create_BirdNET(n_classes, lr=0.01):
    """Creates model, optimizer"""

    model = BirdNET(n_classes)
    model.to(device)
    model.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    return model, optimizer


def create_ResNet50(n_classes, lr=0.01, pretrained=True):
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
                if self.epoch == 0:
                    self.databunch.train_dl.dataset.set_use_cache(True)
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

