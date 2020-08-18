# -*- coding: utf-8 -*-
import numpy as np

X = np.array([1, 2 , 3 , 4], dtype=np.float32)
Y = np.array([3, 6, 9, 12], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()


# gradient
# MSE = 1/N * (w*x - y) ** 2
# dJ/dw = 2x * 1/N * (w*x - y)
def gradient(x, y, y_hat):
    return np.dot(2* x, y_hat - y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training

lr = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction
    y_hat = forward(X)
    
    # loss
    l = loss(Y, y_hat)
    
    # gradients
    dw = gradient(X, Y, y_hat)
    
    # update weeights
    w -= lr * dw
    
    if epoch % 1 == 0:
        print(f'epoch : {epoch + 1}: w = {w: .3f}, loss = {l:.3f}')
    

print(f'Prediction before training: f(5) = {forward(5):.3f}')



