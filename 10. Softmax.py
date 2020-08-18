# -*- coding: utf-8 -*-
import torch
import numpy as np  


# Numpy

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('Softmax numpy: ', outputs)

# Pytorch

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print('Softmax pytoch: ', outputs)