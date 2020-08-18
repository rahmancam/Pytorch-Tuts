# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Option1 - Activation functions as layers
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # nn.Softmax
        # nn.Sigmoid
        # nn.LeakyReLU
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# Option 2 - Activation as functional API
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        # torch.softmax()
        # torch.sigmoid()
        # torch.tanh()
        # torch.relu()
        # F.leaky_relu()
        out = torch.sigmoid(self.linear2(out))
        return out