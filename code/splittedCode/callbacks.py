import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import collections
import re
from helpfuncs import *

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
    def __init__(self, metrics, save_path):
        self.train_stats = Stats(metrics, True)
        self.valid_stats = Stats(metrics, False)
        self.plotCount = 1
        self.save_path

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
            # Check save_path before training to prevent overwriting existing evaluation data!
            save_path = self.save_path / "plots"
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
    def __init__(self, save_path):
        self.best_loss = 10000000.0
        self.save_path
    
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
        }, is_best, filename=self.save_path / "checkpoint.pth.tar")
        
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
        }, self.save_path / "final_epoch_model.pth.tar")
            
        
    def save_checkpoint(self, model_state, is_best, filename="checkpoint.pth.tar"):
        torch.save(model_state, filename)
        if is_best:
            shutil.copyfile(filename, self.save_path / "model_best.pth.tar")   


