from PIL import Image
from torch import nn, optim

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_last=True)

import resnet
import barlow
import datasets


class NYUImageNetDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.6
            ),
            transforms.RandomGrayscale(p=0.2),
            barlow.GaussianBlur(p=0.5),
            barlow.Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = datasets.CustomDataset(root='/dataset', split="train", transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        evalset = datasets.CustomDataset(root='/dataset', split="val", transform=eval_transform)
        eval_loader = torch.utils.data.DataLoader(evalset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        return eval_loader


class ResNetClassifier(LightningModule):
    
    def __init__(self, backboneState, finetuneEpochs):
        super().__init__()
        self.backbone = resnet.get_custom_resnet34()
        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(backboneState)
        
        self.lastLayer = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(1024, 800),
        )

        for layer in self.lastLayer.modules():
           if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.01)
                layer.bias.data.zero_()
        
        self.param_groups = [dict(params=self.lastLayer.parameters(), lr=0.01)]
        self.param_groups.append(dict(params=self.backbone.parameters(), lr=0.0008))
        self.finetuneEpochs = finetuneEpochs
        self.criterion=torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.lastLayer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        classProbs = self.forward(data)
        loss = self.criterion(classProbs, label)
        self.log('train_loss', loss)
        return loss
    
    def _evaluate(self, batch, batch_idx, stage=None):
        x, y = batch
        out = self.forward(x)
        logits = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc
    
    def validation_step(self,batch,batch_idx):
        self._evaluate(batch, batch_idx, 'val')[0]
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.param_groups, 0, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.finetuneEpochs, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def finetuneBackbone(backboneState, finetuneEpochs):
    nyudata = NYUImageNetDataModule()
    classifier = ResNetClassifier(backboneState, finetuneEpochs)

    classifier_trainer = Trainer(gpus=1, deterministic=True, max_epochs=finetuneEpochs, default_root_dir='./classifier',
                        limit_val_batches=1.0, benchmark=True, callbacks=[checkpoint_callback], fast_dev_run=False)

    classifier_trainer.fit(classifier, train_dataloader=nyudata.train_dataloader(), val_dataloaders=nyudata.val_dataloader())
    return classifier.state_dict()