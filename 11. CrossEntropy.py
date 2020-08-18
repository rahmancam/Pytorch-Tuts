# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

# Numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# Y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

# y_pred has probabilites (softmax)
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# Pytorch  
# nn.CrossEntropyLoss applies ->
# nn.LogSoftMax + nn.NLLLoss(negative log likelihood loss)
# -> No softmax in last layer
# Y has class labels (raw class), not one-Hot encoded
# Y_pred has raw scores (logits), no Softmax!
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# n_samples x n_classes = 1 x 3
# class labels -> 0, 1 , 2
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)