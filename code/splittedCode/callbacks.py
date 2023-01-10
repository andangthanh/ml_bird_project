import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import collections
import re
from helpfuncs import *
from settings import device
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns


class Callback():
    _order = 0
    def set_runner(self, run): self.run=run
    
    # delegate object to run object if cant find attribute in Callback class
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')


class UseCacheCallback(Callback):
    _order = 10
    def after_epoch(self):
        if not self.databunch.train_dl.dataset.use_cache:
            self.databunch.train_dl.dataset.set_use_cache(True)
            self.databunch.valid_dl.dataset.set_use_cache(True)


class TestInferenceCallback(Callback):
    _order = 20
    def __init__(self, save_path, LEN_CLASSES, target_names, idx2class):
        self.pred_list = []
        self.true_list = []
        self.save_path = save_path
        self.LEN_CLASSES = LEN_CLASSES
        self.target_names = target_names
        self.idx2class = idx2class

    def after_fit(self):
        print("TEST INFERENCE")
        print(torch.cuda.current_device())
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(self.run.databunch.valid_dl,0):
                xb, yb = batch_data[0].to(device), batch_data[1].to(device)
                y_test_pred = self.model.module(xb)

                y_pred_sig = torch.sigmoid(y_test_pred)
                y_pred_sig = [x.detach().cpu().numpy() for x in y_pred_sig]
                y_pred_tag = np.around(y_pred_sig)

                y_test_tag = [x.detach().cpu().numpy() for x in yb]
                self.pred_list.append(y_pred_tag)
                self.true_list.append(y_test_tag)
                    
        self.pred_list = np.reshape(self.pred_list, (-1,  self.LEN_CLASSES))
        self.true_list = np.reshape(self.true_list, (-1,  self.LEN_CLASSES))

        f1 = f1_score(self.true_list, self.pred_list, average="macro", zero_division=1)
        prec = precision_score(self.true_list, self.pred_list, average="macro", zero_division=1)
        rec = recall_score(self.true_list, self.pred_list, average="macro", zero_division=1)
        acc = accuracy_score(self.true_list, self.pred_list)

        print('f1: ',f1)
        print('precision: ',prec)
        print('recall: ',rec)
        print('accuracy: ',acc)

        report = classification_report(self.true_list, self.pred_list, target_names=self.target_names, output_dict=True, zero_division=1)

        df = pd.DataFrame(report).transpose()
        df.to_csv(self.save_path / "report.csv")
        
        with open(self.save_path / "report.txt", "a") as f:
            print(classification_report(self.true_list, self.pred_list, zero_division=1), file=f)

        # probably needs to be changed later or omitted
        cm = confusion_matrix(np.argmax(self.true_list, axis=1), np.argmax(self.pred_list, axis=1))
        confusion_matrix_df = pd.DataFrame(cm).rename(columns=self.idx2class, index=self.idx2class)
        fig, ax = plt.subplots(figsize=(12,12))         
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax, fmt="d")
        plt.tight_layout()
        fig.savefig(self.save_path / "confusion_matrix.eps", format = 'eps')
        fig.savefig(self.save_path / "confusion_matrix.pdf", format = 'pdf')
        fig.savefig(self.save_path / "confusion_matrix.png", format = 'png')
        plt.close()

        torch.save({
            'pred_list': self.pred_list,
            'true_list': self.true_list,
            'f1': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'report': report,
            'confusion_matrix': cm
        }, self.save_path / "test_inference.pth.tar")




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
    def __init__(self, metrics, save_path, plot_frequency=1, save_plots=True):
        self.train_stats = Stats(metrics, True)
        self.valid_stats = Stats(metrics, False)
        self.plotCount = 0
        self.save_path = save_path
        self.plot_frequency = plot_frequency
        self.save_plots = save_plots 

    def begin_fit(self):
        if self.run.resume:
            self.plotCount = self.run.epoch
    
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

        if self.run.epoch > 0 and self.run.epoch%self.plot_frequency==0 and self.save_plots:
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
            plt.figure(figsize=(6,5))
            plt.title("Loss over epoch No. {}".format(self.plotCount))
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
            plt.figure(figsize=(6,5))
            plt.title("Accuracy over epoch No. {}".format(self.plotCount))
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

            # Plot recall and precision
            plt.figure(figsize=(6,5))
            plt.title("Recall/Precision/F1 over epoch No. {}".format(self.plotCount))
            plt.plot(N, self.train_stats.hist_metrics[2], label = "Training Recall", c='cornflowerblue')
            plt.plot(N, self.valid_stats.hist_metrics[2], label = "Valid Recall", c='aqua')
            plt.plot(N, self.train_stats.hist_metrics[3], label = "Training Precision", c='red')
            plt.plot(N, self.valid_stats.hist_metrics[3], label = "Valid Precision", c='orange')
            plt.plot(N, self.train_stats.hist_metrics[4], label = "Training F1-Score", c='darkgreen')
            plt.plot(N, self.valid_stats.hist_metrics[4], label = "Valid F1-Score", c='limegreen')
            plt.xlabel("Epoch #")
            plt.ylabel("Recall/Precision/F1")
            plt.ylim([0.0, 1.05])
            plt.legend(loc="upper left")
            (save_path / "rec_prec").mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"rec_prec/rec_prec{self.plotCount}.eps", format = 'eps')
            plt.savefig(save_path / f"rec_prec/rec_prec{self.plotCount}.pdf", format = 'pdf')
            plt.savefig(save_path / f"rec_prec/rec_prec{self.plotCount}.png", format = 'png')
            plt.close()
            
        self.plotCount += 1
            
            #plt.show()
    

class SaveCheckpointCallback(Callback):
    _order = 2
    def __init__(self, save_path):
        self.best_loss = 10000000.0
        self.save_path = save_path
    
    def after_epoch(self):
        is_best = self.run.stats.valid_stats.avg_stats[0] < self.best_loss
        self.best_loss = min(self.run.stats.valid_stats.avg_stats[0], self.best_loss)



        # Eventuell nur speichern, wenn bester Loss, jedes mal Checkpoint speichern koennte
        # lange dauern  

        self.save_checkpoint({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_loss_his' : self.run.stats.train_stats.hist_metrics[0],
            'valid_loss_his' : self.run.stats.valid_stats.hist_metrics[0],
            'train_acc' : self.run.stats.train_stats.hist_metrics[1],
            'valid_acc' : self.run.stats.valid_stats.hist_metrics[1],
            'train_macro_rec' : self.run.stats.train_stats.hist_metrics[2],
            'valid_macro_rec' : self.run.stats.valid_stats.hist_metrics[2],
            'train_macro_prec' : self.run.stats.train_stats.hist_metrics[3],
            'valid_macro_prec' : self.run.stats.valid_stats.hist_metrics[3],
            'train_macro_f1' : self.run.stats.train_stats.hist_metrics[4],
            'valid_macro_f1' : self.run.stats.valid_stats.hist_metrics[4],
        }, is_best, filename=self.save_path / "checkpoint.pth.tar")
        
    def after_fit(self):

        torch.save({
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_loss_his' : self.run.stats.train_stats.hist_metrics[0],
            'valid_loss_his' : self.run.stats.valid_stats.hist_metrics[0],
            'train_acc' : self.run.stats.train_stats.hist_metrics[1],
            'valid_acc' : self.run.stats.valid_stats.hist_metrics[1],
            'train_macro_rec' : self.run.stats.train_stats.hist_metrics[2],
            'valid_macro_rec' : self.run.stats.valid_stats.hist_metrics[2],
            'train_macro_prec' : self.run.stats.train_stats.hist_metrics[3],
            'valid_macro_prec' : self.run.stats.valid_stats.hist_metrics[3],
            'train_macro_f1' : self.run.stats.train_stats.hist_metrics[4],
            'valid_macro_f1' : self.run.stats.valid_stats.hist_metrics[4],
        }, self.save_path / "final_epoch_model.pth.tar")
            
        
    def save_checkpoint(self, model_state, is_best, filename="checkpoint.pth.tar"):
        torch.save(model_state, filename)
        if is_best:
            shutil.copyfile(filename, self.save_path / "model_best.pth.tar")   


