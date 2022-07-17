def initialize():
    global FILTERS
    global KERNEL_SIZES
    global RESNET_K
    global RESNET_N
    global LEN_CLASSES
    global device

    FILTERS = [8, 16, 32, 64, 128]
    KERNEL_SIZES = [5, 3, 3, 3, 3]
    RESNET_K = 4
    RESNET_N = 3
    LEN_CLASSES = 15

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())
    print("2")