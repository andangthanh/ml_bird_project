def initialize():
    global FILTERS = [8, 16, 32, 64, 128]
    global KERNEL_SIZES = [5, 3, 3, 3, 3]
    global RESNET_K = 4
    global RESNET_N = 3
    global LEN_CLASSES = 15

    global device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())
    print("2")