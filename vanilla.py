import os
import shutil
import time
import logging
import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from util import *
from model import *
from dataio import *
from config import Config

if  __name__ == '__main__':
    # Backup
    if Config.backup_flag == True:
        Config.pathname = os.path.join('vanilla', Config.dataset, 'priv%d'%(Config.pub_num), Config.date_time_file)
        if not os.path.exists('vanilla'):
            os.makedirs('vanilla')
        if not os.path.exists(os.path.join('vanilla', Config.dataset)):
            os.makedirs(os.path.join('vanilla', Config.dataset))
        if not os.path.exists(os.path.join('vanilla', Config.dataset, 'priv%d'%(Config.pub_num))):
            os.makedirs(os.path.join('vanilla', Config.dataset, 'priv%d'%(Config.pub_num)))
        if not os.path.exists(Config.pathname):
            os.makedirs(Config.pathname)
        if not os.path.exists(os.path.join(Config.pathname, 'ckpt')):
            os.makedirs(os.path.join(Config.pathname, 'ckpt'))
        if not os.path.exists(os.path.join(Config.pathname, 'backup')):
            os.makedirs(os.path.join(Config.pathname, 'backup'))

        shutil.copy('model.py', os.path.join(Config.pathname, 'backup', 'model.py.backup'))
        shutil.copy('vanilla.py', os.path.join(Config.pathname, 'backup', 'vanilla.py.backup'))
        shutil.copy('dataio.py', os.path.join(Config.pathname, 'backup', 'dataio.py.backup'))
        shutil.copy('config.py', os.path.join(Config.pathname, 'backup', 'config.py.backup'))
        shutil.copy('util.py', os.path.join(Config.pathname, 'backup', 'util.py.backup'))
        logging.basicConfig(filename = os.path.join(Config.pathname, 'out.log'))

    torch.manual_seed(Config.manual_seed)
    torch.cuda.manual_seed_all(Config.manual_seed)
    np.random.seed(Config.manual_seed)
    random.seed(Config.manual_seed)

    # Data preparation
    if Config.dataset == 'cifar10':
        if Config.pub_flag == True:
            train_pub_idx= num_split_pub(pub_rand_seed=Config.rand_seed)
        else:
            train_pub_idx= num_split_priv(priv_num=Config.pub_num, priv_rand_seed=Config.rand_seed)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        pub_trainset = DataSet(train_pub_idx, train=True, transform=transform_train)
        pub_trainloader = torch.utils.data.DataLoader(pub_trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)
        glob_testset = datasets.CIFAR10(root=Config.data_path, train=False, download=True, transform=transform_test)
        glob_testloader = torch.utils.data.DataLoader(glob_testset, batch_size=Config.test_batch_size, shuffle=False, num_workers=2)

    elif Config.dataset == 'mnist':
        if Config.pub_flag == True:
            train_pub_idx= num_split_pub(pub_rand_seed=Config.rand_seed)
        else:
            train_pub_idx= num_split_priv(priv_num=Config.pub_num, priv_rand_seed=Config.rand_seed)
        pub_trainset = DataSet(train_pub_idx, train=True)
        pub_trainloader = torch.utils.data.DataLoader(pub_trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)

        glob_testset = datasets.MNIST(Config.data_path, download=True, train=False, transform=transforms.ToTensor())
        glob_testloader = torch.utils.data.DataLoader(glob_testset, batch_size=Config.test_batch_size, shuffle=False, num_workers=2)
    

    if Config.dataset == 'cifar10':
        client_pse = SplitVGG(_model_name = Config.model_name, _out_layer = Config.split_layer).to(Config.device)
        server_priv = SplitVGG(_model_name = Config.model_name, _in_layer = Config.split_layer).to(Config.device)
        top_pse = TransLabels().to(Config.device)
        client_pse.init_w()
        server_priv.init_w()
        top_pse.init_w()

        server_priv.load_state_dict(torch.load('ckpt/cifar10_server_priv.pth'))

        client_pse_opt = torch.optim.SGD(client_pse.parameters(), lr = Config.lr, momentum=0.9, weight_decay=5e-4)
        top_pse_opt = torch.optim.SGD(top_pse.parameters(), lr = Config.lr, momentum=0.9, weight_decay=5e-4)

        client_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_pse_opt, T_max=Config.epochs)
        top_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(top_pse_opt, T_max=Config.epochs)

    elif Config.dataset == 'mnist':
        client_pse = SplitLenet5(_out_layer = Config.split_layer).to(Config.device)
        server_priv = SplitLenet5(_in_layer = Config.split_layer).to(Config.device)
        top_pse = TransLabels().to(Config.device)
        client_pse.init_w()
        server_priv.init_w()
        top_pse.init_w()

        server_priv.load_state_dict(torch.load('ckpt/mnist_server_priv.pth'))

        client_pse_opt = torch.optim.SGD(client_pse.parameters(), lr=Config.lr, momentum=0.05, weight_decay=1e-4)
        top_pse_opt = torch.optim.SGD(top_pse.parameters(), lr = Config.lr, momentum=0.05, weight_decay=1e-4)

        client_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_pse_opt, T_max=4*Config.epochs)
        top_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(top_pse_opt, T_max=4*Config.epochs)

    if Config.criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(Config.epochs * 200):
        start = time.time()
        server_priv.eval()
        client_pse.train()
        top_pse.train()
        train_loss, correct, total= 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(pub_trainloader):
            inputs, targets = inputs.to(Config.device), targets.to(Config.device)
            client_pse_opt.zero_grad()
            top_pse.zero_grad()
            outputs = top_pse(server_priv(client_pse(inputs)))
            loss = criterion(outputs, targets)
            loss.backward()
            client_pse_opt.step()
            top_pse_opt.step()
        
        if (epoch + 1) % 100 == 0:
            acc_glob, loss_glob = test(glob_testloader, top_pse, server_priv, client_pse)
            interval = time.time()-start
            # if acc_glob > best_acc:
            #     print('Saving..')
            #     server_ckpt_path = Config.pathname+'/ckpt/server_%d_%.2f.pth'%(epoch, acc_glob)
            #     client_ckpt_path = Config.pathname+'/ckpt/client_%d_%.2f.pth'%(epoch, acc_glob)
            #     best_acc = acc_glob
            #     torch.save(server_priv.state_dict(), server_ckpt_path)
            #     torch.save(client_priv.state_dict(), client_ckpt_path)
            print(f"{epoch}\t{acc_glob:.3f}%\t{loss_glob:.2f}\t{interval:.2f}")
            logging.critical(f"{epoch}\t{acc_glob:.3f}%\t{loss_glob:.2f}\t{interval:.2f}")
        client_pse_sche.step()
        top_pse_sche.step()
