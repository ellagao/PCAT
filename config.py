import torch
import time

class Config:
    dataset = 'mnist'  # 'mnist', 'cifar10'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Check GPU before running code
    backup_flag = True   # If you don't want to backup current code and output results, you can set it False
    pub_flag = False    # If the dataset of the pseudo-client isn't in the dataset of the real client, pub_flag = True, else False
    date_time_file = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
    pathname = None
    rand_seed = 90
    criterion_name = 'CrossEntropy'
    out_channels = 10
    manual_seed = 10

    if dataset == 'cifar10':

        # Data
        cls_num = 10
        priv_num = 4500
        pub_num = 100
        data_path = './data/cifar10'
        idx_path = './data/cifar10/sampleIdx'

        # Model
        split_layer = 2
        model_name = 'VGG16'
        epochs = 100
        init_w_std = 0.05

        # Training
        lr = 0.1
        loss_flag = 2.1
        train_batch_size = 256
        test_batch_size = 1024

    if dataset == 'mnist':

        # Data
        cls_num = 10
        priv_num = 5400
        pub_num = 10
        data_path = './data/mnist'
        idx_path = './data/mnist/sampleIdx'

        # Model
        split_layer = 2
        model_name = 'lenet5'
        epochs = 15
        init_w_std = 0.1

        # Training
        lr = 0.5
        loss_flag = 2.1
        train_batch_size = 256
        test_batch_size = 1024

