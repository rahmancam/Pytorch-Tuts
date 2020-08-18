# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

class WineDataset(Dataset):
    def __init__(self):
        super(WineDataset, self).__init__()
        xy = np.loadtxt('datasets/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
     
# dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
        
dataset = WineDataset()
batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)


# Training Loop
num_epoch = 2
total_samples = len(dataset)
n_iters = math.ceil(total_samples / batch_size)

for epoch in range(n_iters):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward and backward pass, update
        if (i + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/ {num_epoch}, step {i+1} / {n_iters}, inputs {inputs.shape}')
            
            
    












