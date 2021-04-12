import os
import random
from PIL import Image
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

# VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=800):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,'M'])
        self.fc = nn.Linear(512*6*6,num_classes)
#         self.fcs = nn.Sequential(
#             nn.Linear(512*5*5,4096),
#             nn.ReLU(),
#             nn.Dropout(p = 0.5),
#             nn.Linear(4096,2048),
#             nn.ReLU(),
#             nn.Linear(2048, num_classes)
            
#             )
        self.initialze_weights()
        
    def forward(self, x):
        x = self.conv_layers(x)
#         print("shape: ", x.shape)
        x = x.reshape(x.shape[0],-1)
#         print("shape: ", x.shape)
#         x = self.fcs(x)
        x = self.fc(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if(type(x) == int):
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                    nn.BatchNorm2d(x),
                                    nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
        return nn.Sequential(*layers)
    
    def initialze_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    
    
def get_custom_vgg16():
    x = VGG_net(in_channels = 3, num_classes = 800)
    return x