# -*- coding: utf-8 -*-
# 1. Design model (input size, output size & forward pass)
# 2. Construct loss and optimizer
# 3. Training Loop
#   - Forward pass: Compute prediction and loss
#   - Backward pass - gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        y_hat = self.lin(x)
        return torch.sigmoid(y_hat)
    
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
lr = 0.001
n_iters = 1000

print(f'Features: Input size: {input_dim}, Output size: {output_dim}')

model = LogisticRegression(input_dim, output_dim)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

for epoch in range(n_iters):
    # Forward pass
    y_hat = model(X_train)
    loss = criterion(y_hat, y_train)
    
    # Backward
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss= {loss.item():.4f}')
        
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy : {accuracy:.4f}')
     

        
    



