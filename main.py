import os
import shutil
import time
import logging

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
        Config.pathname = os.path.join('results', Config.dataset, 'priv%d'%(Config.pub_num), Config.date_time_file)
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists(os.path.join('results', Config.dataset)):
            os.makedirs(os.path.join('results', Config.dataset))
        if not os.path.exists(os.path.join('results', Config.dataset, 'priv%d'%(Config.pub_num))):
            os.makedirs(os.path.join('results', Config.dataset, 'priv%d'%(Config.pub_num)))
        if not os.path.exists(Config.pathname):
            os.makedirs(Config.pathname)
        if not os.path.exists(os.path.join(Config.pathname, 'ckpt')):
            os.makedirs(os.path.join(Config.pathname, 'ckpt'))
        if not os.path.exists(os.path.join(Config.pathname, 'backup')):
            os.makedirs(os.path.join(Config.pathname, 'backup'))

        shutil.copy('model.py', os.path.join(Config.pathname, 'backup', 'model.py.backup'))
        shutil.copy('main.py', os.path.join(Config.pathname, 'backup', 'main.py.backup'))
        shutil.copy('dataio.py', os.path.join(Config.pathname, 'backup', 'dataio.py.backup'))
        shutil.copy('config.py', os.path.join(Config.pathname, 'backup', 'config.py.backup'))
        shutil.copy('util.py', os.path.join(Config.pathname, 'backup', 'util.py.backup'))
        logging.basicConfig(filename = os.path.join(Config.pathname, 'out.log'))

    torch.manual_seed(Config.manual_seed)
    torch.cuda.manual_seed_all(Config.manual_seed)
    random.seed(Config.manual_seed)

    # Data preparation
    if Config.dataset == 'cifar10':
        if Config.pub_flag == True:
            train_pub_idx= num_split_pub(pub_rand_seed=Config.rand_seed)
        else:
            train_pub_idx= num_split_priv(priv_num=Config.pub_num, priv_rand_seed=Config.rand_seed)
        train_priv_idx = num_split_priv()
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        pub_trainset = DataSet(train_pub_idx, train=True, transform=transform_train)
        priv_trainset = DataSet(train_priv_idx, train=True, transform=transform_train)
        pub_trainloader = cls_random_loader(pub_trainset)   # label alignment
        priv_trainloader = torch.utils.data.DataLoader(priv_trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)
        glob_testset = datasets.CIFAR10(root=Config.data_path, train=False, download=True, transform=transform_test)
        glob_testloader = torch.utils.data.DataLoader(glob_testset, batch_size=Config.test_batch_size, shuffle=False, num_workers=2)

    elif Config.dataset == 'mnist':
        if Config.pub_flag == True:
            train_pub_idx= num_split_pub(pub_rand_seed=Config.rand_seed)
        else:
            train_pub_idx= num_split_priv(priv_num=Config.pub_num, priv_rand_seed=Config.rand_seed)
        train_priv_idx = num_split_priv()

        pub_trainset = DataSet(train_pub_idx, train=True)
        priv_trainset = DataSet(train_priv_idx, train=True)
        pub_trainloader = cls_random_loader(pub_trainset)
        priv_trainloader = torch.utils.data.DataLoader(priv_trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)

        glob_testset = datasets.MNIST(Config.data_path, download=True, train=False, transform=transforms.ToTensor())
        glob_testloader = torch.utils.data.DataLoader(glob_testset, batch_size=Config.test_batch_size, shuffle=False, num_workers=2)
    

    if Config.dataset == 'cifar10':
        client_pse = SplitVGG(_model_name = Config.model_name, _out_layer = Config.split_layer).to(Config.device)
        client_priv = SplitVGG(_model_name = Config.model_name, _out_layer = Config.split_layer).to(Config.device)
        server_priv = SplitVGG(_model_name = Config.model_name, _in_layer = Config.split_layer).to(Config.device)
        client_pse.init_w()
        client_priv.init_w()
        server_priv.init_w()

        client_pse_opt = torch.optim.SGD(client_pse.parameters(), lr = Config.lr, momentum=0.9, weight_decay=5e-4)
        client_priv_opt = torch.optim.SGD(client_priv.parameters(), lr = Config.lr, momentum=0.9, weight_decay=5e-4)
        server_priv_opt = torch.optim.SGD(server_priv.parameters(), lr = Config.lr, momentum=0.9, weight_decay=5e-4)
        client_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_pse_opt, T_max=Config.epochs)
        client_priv_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_priv_opt, T_max=Config.epochs)
        server_priv_sche = torch.optim.lr_scheduler.CosineAnnealingLR(server_priv_opt, T_max=Config.epochs)

    elif Config.dataset == 'mnist':
        client_pse = SplitLenet5(_out_layer = Config.split_layer).to(Config.device)
        client_priv = SplitLenet5(_out_layer = Config.split_layer).to(Config.device)
        server_priv = SplitLenet5(_in_layer = Config.split_layer).to(Config.device)
        client_pse.init_w()
        client_priv.init_w()
        server_priv.init_w()

        client_pse_opt = torch.optim.SGD(client_pse.parameters(), lr=Config.lr, momentum=0.05, weight_decay=1e-4)
        client_priv_opt = torch.optim.SGD(client_priv.parameters(), lr=Config.lr, momentum=0.05, weight_decay=1e-4)
        server_priv_opt = torch.optim.SGD(server_priv.parameters(), lr=Config.lr, momentum=0.05, weight_decay=1e-4)
        client_pse_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_pse_opt, T_max=4*Config.epochs)
        client_priv_sche = torch.optim.lr_scheduler.CosineAnnealingLR(client_priv_opt, T_max=4*Config.epochs)
        server_priv_sche = torch.optim.lr_scheduler.CosineAnnealingLR(server_priv_opt, T_max=4*Config.epochs)

    if Config.criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(Config.epochs):
        start = time.time()
        loss_priv = Config.loss_flag + 1
        for batch_idx, (inputs, targets) in enumerate(priv_trainloader):
            if loss_priv < Config.loss_flag:
                server_priv_opt.zero_grad()
                server_priv.eval()
                client_pse.train()
                images_pse, labels_pse = pub_trainloader.get_batch(targets)
                images_pse, labels_pse = images_pse.to(Config.device), labels_pse.to(Config.device)
                client_pse_opt.zero_grad()
                pred_pse = server_priv(client_pse(images_pse))
                loss_pse = criterion(pred_pse, labels_pse)
                loss_pse_cp = loss_pse.clone().item()
                loss_pse.backward()
                client_pse_opt.step()
            
            inputs, targets = inputs.to(Config.device), targets.to(Config.device)
            client_priv.train()
            server_priv.train()
            client_priv_opt.zero_grad()
            server_priv_opt.zero_grad()
            pred_priv = server_priv(client_priv(inputs))
            loss = criterion(pred_priv, targets)
            loss_priv=loss.clone().item()
            loss.backward()
            server_priv_opt.step()
            client_priv_opt.step()

        else:
            acc_priv_glob, loss_priv_glob = test(glob_testloader,server_priv, client_priv)
            acc_pse_glob, loss_pse_glob = test(glob_testloader, server_priv, client_pse)
            interval = time.time() - start
            print(f"{epoch}\t{(acc_priv_glob):.2f}%\t{loss_priv_glob:.3f}\t{(acc_pse_glob):.2f}%\t{loss_pse_glob:.3f}\t{interval:.1f}")

            logging.critical(f"{epoch}\t{(acc_priv_glob):.2f}%\t{loss_priv_glob:.3f}\t{(acc_pse_glob):.2f}%\t{loss_pse_glob:.3f}\t{interval:.1f}")

            # Save ckpt
            if epoch > (Config.epochs - 10) and Config.backup_flag == True:
                client_priv_ckpt_path=Config.pathname+'/ckpt/client_priv_e%d_%.2f_%.2f.pth'%(epoch,acc_priv_glob, acc_pse_glob)
                torch.save(client_priv.state_dict(), client_priv_ckpt_path)
                server_priv_ckpt_path=Config.pathname+'/ckpt/server_priv_e%d_%.2f_%.2f.pth'%(epoch,acc_priv_glob, acc_pse_glob)
                torch.save(server_priv.state_dict(), server_priv_ckpt_path)
                client_pse_ckpt_path=Config.pathname+'/ckpt/client_pse_e%d_%.2f_%.2f.pth'%(epoch,acc_priv_glob, acc_pse_glob)
                torch.save(client_pse.state_dict(), client_pse_ckpt_path)

        server_priv_sche.step()
        client_priv_sche.step()
        client_pse_sche.step()

