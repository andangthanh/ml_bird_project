import torch

FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [5, 3, 3, 3, 3]
RESNET_K = 4
RESNET_N = 3
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
