import torch
import random
import math
from config import Config
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

def num_split_pub(is_train=True, pub_num=Config.pub_num, pub_rand_seed=0):
    pub = torch.empty(0)
    if is_train: idx_root = Config.idx_path + '/shuffle/train_glob'
    else: idx_root = Config.idx_path + '/shuffle/test_glob'
    for cls in range(Config.cls_num):
        tmp_all = torch.load(idx_root+str(cls)+'.dat')
        pub = torch.cat([pub,tmp_all[(-(pub_num + pub_rand_seed)) : (-pub_rand_seed)]], 0)
    return pub

def num_split_priv(is_train=True, priv_num=Config.priv_num, priv_rand_seed=0):
    priv = torch.empty(0)
    if is_train: idx_root=Config.idx_path + '/shuffle/train_glob'
    else: idx_root=Config.idx_path + '/shuffle/test_glob'
    for cls in range(Config.cls_num):
        tmp_all = torch.load(idx_root+str(cls)+'.dat')
        priv = torch.cat([priv,tmp_all[priv_rand_seed: (priv_num + priv_rand_seed)]],0)
    return priv

class DataSet(Dataset):
    def __init__(self, idx_tensor, train=True, transform=transforms.ToTensor(), target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.idx_tensor = idx_tensor
        if Config.dataset == 'cifar10':
            self.torch_dataset = datasets.CIFAR10(root=Config.data_path, train=train, download=True, transform=self.transform,target_transform=self.target_transform)
        elif Config.dataset == 'mnist':
            self.torch_dataset = datasets.MNIST(root=Config.data_path, train=train, download=True, transform=transform,target_transform=target_transform)
            
    def __len__(self):
        return self.idx_tensor.shape[0]

    def __getitem__(self, idx):
        real_idx=self.idx_tensor[idx]
        real_idx=real_idx.long()
        image, label = self.torch_dataset[real_idx] # image shape: torch.Size([1, height, width])
        return image, label

class cls_random_loader():
    def __init__(self, dataset):
        self.dataset = dataset

    def get_batch(self, labels):
        cls_num_batch = torch.zeros((Config.cls_num)).int().to(Config.device)
        for cls_idx in range(Config.cls_num):
            cls_mask = (labels == cls_idx)
            cls_num_batch[cls_idx] = cls_mask.type(torch.int).sum().item()
        cls_num_batch = cls_num_batch // 2
        out_num = cls_num_batch.sum().item()
        out_label = torch.zeros(out_num).long()
        if Config.dataset == 'cifar10':
            out_batch = torch.zeros((out_num, 3, 32, 32))
        elif Config.dataset == 'mnist':
            out_batch = torch.zeros((out_num, 1, 28, 28))

        cur_idx = 0
        for cls_idx in range(Config.cls_num):
            num_per_cls = cls_num_batch[cls_idx].item()
            for sam_idx in range(num_per_cls):
                random_part = random.randint(0, Config.pub_num - 1)
                sam_pos = cls_idx * Config.pub_num + random_part
                out_batch[cur_idx], out_label[cur_idx] = self.dataset[sam_pos]
                cur_idx += 1
        return out_batch, out_label
    
