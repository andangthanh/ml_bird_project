import torch
import random
import torch.nn as nn

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


def target_to_one_hot(num_class, target):
    NUM_CLASS = num_class
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


class DownmixMono(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform):
        return torch.mean(waveform, dim=0, keepdim=True)


class ToDecibels(nn.Module):
    def __init__(self,
                 power=2, # magnitude=1, power=2
                 ref='max',
                 top_db=80,
                 normalized=True,
                 amin=1e-7):
        super().__init__()
        self.constant = 10.0 if power == 2 else 20.0
        self.ref = ref
        self.top_db = abs(top_db) if top_db else top_db
        self.normalized = normalized
        self.amin = amin

    def forward(self, x):
        batch_size = x.shape[0]
        if self.ref == 'max':
            ref_value = x.contiguous().view(batch_size, -1).max(dim=-1)[0]  #max value per sample
            ref_value.unsqueeze_(1).unsqueeze_(1) #reshape in form of [batchsize,1,1]
        else:
            ref_value = torch.tensor(self.ref)
        spec_db = x.clamp_min(self.amin).log10_().mul_(self.constant)
        spec_db.sub_(ref_value.clamp_min_(self.amin).log10_().mul_(10.0))
        if self.top_db is not None:
            max_spec = spec_db.reshape(batch_size, -1).max(dim=-1)[0]
            max_spec.unsqueeze_(1).unsqueeze_(1)
            spec_db = torch.max(spec_db, max_spec - self.top_db)
            if self.normalized:
                # normalize to [0, 1]
                spec_db.add_(self.top_db).div_(self.top_db)
        return spec_db
