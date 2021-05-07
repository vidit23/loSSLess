from PIL import Image
from torch import nn, optim

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy

from tqdm import tqdm

import resnet
import barlow
import datasets


unlabeled_train_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    ),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.2, hue=0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    barlow.GaussianBlur(p=0.5),
    barlow.Solarization(p=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_dataset():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        barlow.GaussianBlur(p=0.5),
        barlow.Solarization(p=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = datasets.CustomDatasetWithIndex(root='/dataset', split="train", transform=train_transform)
    return trainset



class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet.get_custom_resnet34()
        self.backbone.fc = nn.Identity()
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
        self.param_groups.append(dict(params=self.backbone.parameters(), lr=0.0005))
        self.criterion=torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.lastLayer(x)
        return x



def unlabeledActiveLearning(finetunedModel):
    
    classifier = ResNetClassifier()
    classifier.load_state_dict(finetunedModel)
    classifier = classifier.cuda()

    evalset = datasets.CustomDatasetWithIndex(root='/dataset', split="val", transform=normalize_transform)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=512, shuffle=True, num_workers=4)
    entireUnlabeledDataset = datasets.CustomDatasetWithIndex(root='/dataset', split="unlabeled", transform=normalize_transform)
    toBeRankedIndices = torch.tensor([i for i in range(len(entireUnlabeledDataset))])

    criterion = nn.CrossEntropyLoss()
    learning = 0.0001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2, verbose=True)  

    originalAndExtraDataset = train_dataset()

    activeLearningLoopCount = 3
    skimTopPercentage = 3

    for i in range(activeLearningLoopCount):
        print("\n\nRunning loop number", i)
        unlabeledFilteredData = torch.utils.data.Subset(entireUnlabeledDataset, toBeRankedIndices.tolist())
        unlabeledFiteredDataLoader = torch.utils.data.DataLoader(unlabeledFilteredData, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        allConfidenceScores, predictedLabels = torch.Tensor(), torch.Tensor()
        actualLabels, allIndices, allImageTensors = torch.Tensor(), torch.tensor([]), torch.Tensor()
        
        classifier.eval()
        print("\tStarting the evaluation process with unlabeled data")
        with torch.no_grad():
            # Going through the left over unlabeled set and collecting the confidence for model predictions
            numOfBatches = len(unlabeledFilteredData) / unlabeledFiteredDataLoader.batch_size
            for idx, batch in tqdm(enumerate(unlabeledFiteredDataLoader), total=int(numOfBatches)):
                images, _, indices = batch
                images = images.cuda()
                classScores = classifier(images)
                classLogits = F.softmax(classScores, dim=1)
                
                labelConfidence, predictions = torch.max(classLogits.data, 1)
                
                sortedBatchConfidence, sortedBatchConfidencePos = torch.sort(labelConfidence, descending=True)
                topConfidencePos = sortedBatchConfidencePos[:150]
                
                allConfidenceScores = torch.cat((allConfidenceScores, labelConfidence[topConfidencePos].cpu()))
                predictedLabels = torch.cat((predictedLabels, predictions[topConfidencePos].cpu()))
                allIndices = torch.cat((allIndices, indices[topConfidencePos].cpu()))
                allImageTensors = torch.cat((allImageTensors, images[topConfidencePos].cpu()))

            print("\tGot the predictions of" , len(unlabeledFilteredData), " images")

            # Sorting all the predictions based on the confidence scores and the argsort
            sortedConfidence, sortedConfidencePos = torch.sort(allConfidenceScores, descending=True)
            print("\tSorted the predictions based on confidence scores")

            # Calculating how many top predictions to retrain the model on
            limit = int(len(unlabeledFilteredData) * (skimTopPercentage/100))
            topConfidencePos = sortedConfidencePos[:limit]
            print("\tGot the top ", limit, "confidence indices")
            skimTopPercentage -= 1
            

            # Fetching the top confidence's index in original dataset
            topConfidenceIndices = allIndices[topConfidencePos]
            # And removing these indices from toBeRankedIndices
            combined = torch.cat((toBeRankedIndices, topConfidenceIndices))
            uniques, counts = combined.unique(return_counts=True)
            toBeRankedIndices = uniques[counts == 1]
            print("\tRemoved the indices of top ranked from further consideration")
            
            # Fetching the top confidence's images and labels
            topConfidenceImages = allImageTensors[topConfidencePos]
            topConfidenceLabels = predictedLabels[topConfidencePos]
            additionalTopConfidenceDataset = datasets.CustomTensorDataset((topConfidenceImages, topConfidenceLabels, topConfidenceIndices), unlabeled_train_transform)
            originalAndExtraDataset = torch.utils.data.ConcatDataset((additionalTopConfidenceDataset, originalAndExtraDataset))
            
            originalAndExtraTopConfidenceDataLoader = torch.utils.data.DataLoader(originalAndExtraDataset, batch_size=128, shuffle=True,num_workers=4, pin_memory=True)
            print("\tCombined the original training set and the new dataset to a length of", len(originalAndExtraDataset))

        numOfBatches = len(originalAndExtraDataset) / originalAndExtraTopConfidenceDataLoader.batch_size
        print("\tStarting to train the model")
        for epoch in range(15):
            classifier.train()
            running_loss = 0.0
            for idx, data in tqdm(enumerate(originalAndExtraTopConfidenceDataLoader), total=int(numOfBatches)):
                inputs, labels, idx = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = classifier(inputs)
                loss = criterion(outputs, labels.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in evalloader:
                    images, labels, idx = data

                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = classifier(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = (100 * correct / total)
            
            scheduler.step(accuracy)
            learning = scheduler._last_lr[0]
            print("\t\tLoss at epoch", epoch, "is", running_loss/numOfBatches)
            print("\t\t Current learning rate", learning)
            print(f"\t\tTeam 15: Accuracy: {accuracy:.2f}%")

    return classifier.state_dict()
