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
