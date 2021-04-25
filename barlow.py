from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

import lightly
import lightly.models as models
import lightly.loss as loss
import lightly.data as data
from lightly.models.barlowtwins import BarlowTwins
from lightly.models.simclr import SimCLR

from simclr.modules.transformations import TransformsSimCLR
from PIL import Image, ImageOps, ImageFilter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, limit=0):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        if limit == 0:
            self.num_images = len(os.listdir(self.image_dir))
        else:
            self.num_images = limit

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform == None:
            return img, self.labels[idx]            

        return self.transform(img), self.labels[idx]
    

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    

    
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1



class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [512] + list(map(int, '1024-1024-1024'.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(1024)
#         torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        loss = on_diag + 3.9e-3 * off_diag
        return loss
    
    

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adjust_learning_rate(optimizer, loader, step):
    max_steps = 1000 * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = 0.2 * 1024 / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



# create a dataset from your image folder
dataset = CustomDataset(root='/dataset', split='unlabeled', transform=Transform())

# build a PyTorch dataloader
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=4)

torch.backends.cudnn.benchmark = True

model = BarlowTwins().cuda()
# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
optimizer = LARS(model.parameters(), lr=0, weight_decay=1e-6,
                 weight_decay_filter=exclude_bias_and_norm,
                 lars_adaptation_filter=exclude_bias_and_norm)

# automatically resume from checkpoint if it exists, RELOAD MODEL
if os.path.isfile('/scratch/vvb238/barlow-34-1000/checkpoint.pth'):
    ckpt = torch.load('/scratch/vvb238/barlow-34-1000/checkpoint.pth',
                      map_location='cpu')
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
else:
    start_epoch = 0
    

    
least_loss = float('inf')
running_loss = 0

start_time = time.time()
scaler = torch.cuda.amp.GradScaler()
for epoch in range(start_epoch, 1000):
#     sampler.set_epoch(epoch)
    for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
        y1 = y1.cuda()
        y2 = y2.cuda()
        lr = adjust_learning_rate(optimizer, loader, step)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model.forward(y1, y2)
        
        running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 10 == 0:
            stats = dict(epoch=epoch, step=step, learning_rate=lr,
                         loss=loss.item(),
                         time=int(time.time() - start_time))
            print(json.dumps(stats), flush=True)
#                 print(json.dumps(stats), file=stats_file)
        # save checkpoint
    state = dict(epoch=epoch + 1, model=model.state_dict(),
                 optimizer=optimizer.state_dict())
    if running_loss < least_loss:
        least_loss = running_loss
        torch.save(state, '/scratch/vvb238/barlow-34-1000/best-checkpoint.pth')
    running_loss = 0
    # SAVE MODEL AFTER EVERY EPOCH
    torch.save(state, '/scratch/vvb238/barlow-34-1000/checkpoint.pth')
    
# FINAL MODEL SAVING
torch.save(model.backbone.state_dict(),
           '/scratch/vvb238/barlow-34-1000/resnet50.pth')
