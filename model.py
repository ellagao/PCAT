'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from config import Config

cfg_split = {
    'VGG11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'VGG19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],}

class SplitVGG(nn.Module):
    def __init__(self, _model_name, _in_layer=0, _out_layer=6):
        super(SplitVGG, self).__init__()
        self.model_name = _model_name
        self.in_layer = _in_layer
        self.out_layer = _out_layer
        if self.in_layer not in [0, 1, 2, 3, 4, 5]:
            raise ValueError('Invalid in layer.')             
        if self.out_layer not in [1, 2, 3, 4, 5, 6]:
            raise ValueError('Invalid out layer.')
        if self.out_layer <= self.in_layer:
            raise ValueError('Invalid in layer and out layer: out layer should be larger than in layer.')

        if self.in_layer == 0:
            self.layer1 = self._make_layers(cfg_split[self.model_name][0], 3)
        if self.in_layer <= 1 and self.out_layer >= 2: 
            self.layer2 = self._make_layers(cfg_split[self.model_name][1], 64)
        if self.in_layer <= 2 and self.out_layer >= 3: 
            self.layer3 = self._make_layers(cfg_split[self.model_name][2],128)
        if self.in_layer <= 3 and self.out_layer >= 4: 
            self.layer4 = self._make_layers(cfg_split[self.model_name][3],256)
        if self.in_layer <= 4 and self.out_layer >= 5: 
            self.layer5 = self._make_layers(cfg_split[self.model_name][4],512)
        if self.out_layer == 6:
            self.AvgPool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.classifier = nn.Linear(512, Config.cls_num)

    def forward(self, x, noise_flag=False):
        if noise_flag == True and Config.noise_std != 0:
            temp_size = x.size()
            x = x + torch.normal(mean = Config.noise_mean, std = Config.noise_std, size = temp_size).to(Config.device)
        if self.in_layer == 0: x = self.layer1(x)
        if self.out_layer == 1: return x
        if self.in_layer in [0, 1]: x = self.layer2(x)
        if self.out_layer == 2: return x
        if self.in_layer in [0, 1, 2]: x = self.layer3(x)
        if self.out_layer == 3: return x
        if self.in_layer in [0, 1, 2, 3]: x = self.layer4(x)
        if self.out_layer == 4: return x
        if self.in_layer in [0, 1, 2, 3, 4]: x = self.layer5(x)
        if self.out_layer == 5: return x
        x = self.AvgPool(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
    
    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=Config.init_w_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


class SplitLenet5(nn.Module):
    def __init__(self, _in_layer=0, _out_layer=5):
        super(SplitLenet5, self).__init__()

        self.in_layer = _in_layer
        self.out_layer = _out_layer
        if self.in_layer not in [0, 1, 2, 3, 4]:
            raise ValueError('Invalid in layer.')             
        if self.out_layer not in [1, 2, 3, 4, 5]:
            raise ValueError('Invalid out layer.')
        if self.out_layer <= self.in_layer:
            raise ValueError('Invalid in layer and out layer: out layer should be larger than in layer.')

        if self.in_layer == 0:
            self.layer1 = nn.Sequential(
                          nn.Conv2d(1, 6, kernel_size=5,padding=2), 
                          nn.ReLU(False), 
                          nn.MaxPool2d(kernel_size=2, stride=2))
        if self.in_layer <= 1 and self.out_layer >= 2: 
            self.layer2 = nn.Sequential(
                          nn.Conv2d(6, 16, kernel_size=5), 
                          nn.ReLU(False), 
                          nn.MaxPool2d(kernel_size=2, stride=2))
        if self.in_layer <= 2 and self.out_layer >= 3:
            self.layer3 = nn.Sequential(
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), 
                          nn.Sigmoid())
        if self.in_layer <= 3 and self.out_layer >= 4:
            self.layer4 = nn.Sequential(
                          nn.Linear(120, 84), 
                          nn.Sigmoid())
        if self.out_layer == 5:
            self.layer5 = nn.Linear(84, Config.out_channels)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=Config.init_w_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x, noise_flag=False):
        if noise_flag == True:
            temp_size = x.size()
            x = x + torch.normal(mean = Config.noise_mean, std = Config.noise_std, size = temp_size).to(Config.device)
        if self.in_layer == 0: x = self.layer1(x)
        if self.out_layer == 1: return x
        if self.in_layer in [0, 1]: x = self.layer2(x)
        if self.out_layer == 2: return x
        if self.in_layer in [0, 1, 2]: x = self.layer3(x)
        if self.out_layer == 3: return x
        if self.in_layer in [0, 1, 2, 3]: x = self.layer4(x)
        if self.out_layer == 4: return x
        x = self.layer5(x)
        return x
    
