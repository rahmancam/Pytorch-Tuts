# -*- coding: utf-8 -*-
# 1. Design model - input_size, output_size, forward pass
# 2. Construct loss and optimizer
# 3. Training Loop
#       - Forward pass - compute prediction and loss
#       - Backward pass- Gradients
#       - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

print(f'X shape: {X.shape}, y shape: {y.shape}')

y = y.view(y.shape[0], 1)

print(f'X shape: {X.shape}, y shape: {y.shape}')

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features
                                            
# define model

lr = 0.001
n_iters = 1000

model = nn.Linear(input_size, output_size)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training loop
for epoch in range(n_iters):
    # forward pass
    y_hat = model(X)
    # loss
    l = criterion(y_hat, y)
    
    #backward pass
    l.backward()
    
    # update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {l.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()




