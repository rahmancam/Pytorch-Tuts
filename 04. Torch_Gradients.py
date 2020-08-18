# -*- coding: utf-8 -*-
import torch

X = torch.tensor([1, 2 , 3 , 4], dtype=torch.float32)
Y = torch.tensor([3, 6, 9, 12], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training

lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction
    y_hat = forward(X)
    
    # loss
    l = loss(Y, y_hat)
    
    # gradients
    l.backward() # dl/dw
    
    # update weeights
    with torch.no_grad():
        w -= lr * w.grad
        
    # clear gradients
    w.grad.zero_()
    
    
    if epoch % 1 == 0:
        print(f'epoch : {epoch + 1}: w = {w: .3f}, loss = {l:.3f}')
    

print(f'Prediction before training: f(5) = {forward(5):.3f}')



