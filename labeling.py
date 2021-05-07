from PIL import Image
from torch import nn, optim

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy

from tqdm import tqdm

from sklearn.cluster import KMeans
import numpy as np

import resnet
import barlow
import datasets


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


def generateLabels():
    classifier = ResNetClassifier()

    if os.path.isfile('./model.pth'):
        ckpt = torch.load('./model.pth', map_location='cpu')
        classifier.load_state_dict(ckpt)

    unlabeled_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    entireUnlabeledDataset = datasets.CustomDatasetWithIndex(root='/dataset', split="unlabeled", transform=unlabeled_transform)

    classifier = classifier.cuda()
    entireUnlabeledDataLoader = torch.utils.data.DataLoader(entireUnlabeledDataset, batch_size=512, 
                                                            shuffle=True, num_workers=4, pin_memory=True)



    allDifferenceInTopTwo, predictedLabels = torch.Tensor(), torch.Tensor()
    actualLabels, allIndices, allImageTensors = torch.Tensor(), torch.tensor([]), torch.Tensor()

    classifier.eval()
    print("\tStarting the evaluation process with unlabeled data")
    with torch.no_grad():
        # Going through the left over unlabeled set and collecting the confidence for model predictions
        numOfBatches = len(entireUnlabeledDataset) / entireUnlabeledDataLoader.batch_size
        for idx, batch in tqdm(enumerate(entireUnlabeledDataLoader), total=int(numOfBatches)):
            images, labels, indices = batch

            images = images.cuda()
    #             labels = labels.cuda()

            classScores = classifier(images)
            classLogits = F.softmax(classScores, dim=1)

            # Taking the top 2 values in the class prediction for each image
            labelConfidence, predictions = torch.sort(classLogits.data, dim=1, descending=True)
            # And subtracting those values
            differenceInTopTwo = labelConfidence[:, 0] - labelConfidence[:, 1]
            # Sorting based on the subtracted values, this gives the images with most confusion between top two classes
            sortedDifferenceInTopTwo, sortedDifferenceInTopTwoPos = torch.sort(differenceInTopTwo, descending=False)
            # Taking the top 150 of the confusion to avoid memory overload
            topSortedDifferenceInTopTwoPos = sortedDifferenceInTopTwoPos[:150]

            allDifferenceInTopTwo = torch.cat((allDifferenceInTopTwo, differenceInTopTwo[topSortedDifferenceInTopTwoPos].cpu()))
            allIndices = torch.cat((allIndices, indices[topSortedDifferenceInTopTwoPos].cpu()))
            

        print("\tGot the predictions of" , len(entireUnlabeledDataset), " images")

        # Sorting all the predictions based on the confidence scores and the argsort
        allSortedDifferenceInTopTwo, allSortedDifferenceInTopTwoPos = torch.sort(allDifferenceInTopTwo, descending=False)
        print("\tSorted the predictions based on confidence scores")

        # Calculating how many top predictions to retrain the model on
        leastDifferenceInTopTwoPos = allSortedDifferenceInTopTwoPos[:100000]
        print("\tGot the top ", 100000, "confidence indices")


        # Fetching the top confidence's index in original dataset
        topConfidenceIndices = allIndices[leastDifferenceInTopTwoPos]



    unlabeledFilteredData = torch.utils.data.Subset(entireUnlabeledDataset, topConfidenceIndices.tolist())
    unlabeledFiteredDataLoader = torch.utils.data.DataLoader(unlabeledFilteredData, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)


    allIndices, allImageEncoding = torch.Tensor(), torch.tensor([])

    classifier.eval()
    print("\tStarting the evaluation process with unlabeled data")
    with torch.no_grad():
        # Going through the left over unlabeled set and collecting the confidence for model predictions
        numOfBatches = len(unlabeledFilteredData) / unlabeledFiteredDataLoader.batch_size
        for idx, batch in tqdm(enumerate(unlabeledFiteredDataLoader), total=int(numOfBatches)):
            images, labels, indices = batch

            images = images.cuda()
    #             labels = labels.cuda()

            classScores = classifier(images)
            classLogits = F.softmax(classScores, dim=1)

            allImageEncoding = torch.cat((allImageEncoding, classLogits.cpu()))
            allIndices = torch.cat((allIndices, indices.cpu()))

    allImageEncoding = allImageEncoding.numpy()
    allIndices = allIndices.numpy()

    kmeans = KMeans(n_clusters=800, n_init=5, verbose=1).fit(allImageEncoding)


    from collections import defaultdict
    clusterImageIdMap = defaultdict(list)
    totalCount = 0
    for clusterId, image in zip(kmeans.labels_, allIndices):
        if len(clusterImageIdMap[clusterId]) < 17:
            clusterImageIdMap[clusterId].append(image)
            totalCount += 1
        if totalCount == 12800:
            print("Reached max limit")
            break

    f = open("imageRequest-23.txt", "a")
    for key in clusterImageIdMap:
        for image in clusterImageIdMap[key]:
            f.write(str(int(image)) + ".png\n")
    f.close()