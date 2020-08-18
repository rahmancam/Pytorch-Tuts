# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WineDataset(Dataset):
    
    def __init__(self, transform=None):
        xy = np.loadtxt('datasets/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.transform = transform
        
        # We wil convert the samples to tensors via transforms
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
    
dataset = WineDataset(transform=ToTensor())
firstdata = dataset[0]
features, labels = firstdata
print(features)
print(type(features), type(labels))

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

composed = transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)
firstdata = dataset[0]
features, labels = firstdata
print(features)
print(type(features), type(labels))

