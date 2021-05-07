import os
import torch
from torch.utils.data import Dataset

from PIL import Image, ImageOps, ImageFilter



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


class CustomDatasetWithIndex(torch.utils.data.Dataset):
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
            self.labels = torch.load(label_path).float()
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        idx = int(idx)
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img), self.labels[idx], torch.tensor(idx).float()



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.unlabeled_train_transform = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.unlabeled_train_transform:
            x = self.unlabeled_train_transform(x)

        y = self.tensors[1][index]

        return x, y, self.tensors[2][index]

    def __len__(self):
        return self.tensors[0].size(0)