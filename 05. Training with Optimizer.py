# -*- coding: utf-8 -*-
# Pipeline
# 1. Design model - (input_size, output_size, foward_pass)
# 2. Construct loss and optimizer
# 3. Training Loop
#       - Forward pass - compute prediction
#       - Backward pass - gradients
#       - Update weights

import torch
import torch.nn as nn

X = torch.tensor([[1], [2] , [3] , [4]], dtype=torch.float32)
Y = torch.tensor([[3], [6], [9], [12]], dtype=torch.float32)


X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features


# model = nn.Linear(input_size, output_size)

# Alternative- custom
class CustomLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)

model = CustomLinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training

lr = 0.01
n_iters = 500

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(n_iters):
    # prediction
    y_hat = model(X)
    
    # loss
    l = loss(Y, y_hat)
    
    # gradients
    l.backward() # dl/dw
    
    # update weeights
    optimizer.step()
    
    # clear gradients
    optimizer.zero_grad()
    
    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch : {epoch + 1}: w = {w[0][0].item(): .3f}, loss = {l:.3f}')
    

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



