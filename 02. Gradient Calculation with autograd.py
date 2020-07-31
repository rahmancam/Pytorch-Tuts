# -*- coding: utf-8 -*-

# Gradients are essential for model optimization
import torch

# (requires gradient)
x = torch.randn(3, requires_grad=True)
print(x)

# Forward pass
# Pytorch automaticaly created grad_fn for us
y = x + 2 # creates a computational graph
print(y) # AddBackward graph
z = y * y * 2
print(z) # MulBackward graph

# if z is not scalar
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad)

# # z is scalar
# z = z.mean()
# print(z) # MeanBackward graph
# z.backward() # dz/dx (calcualted using chain rules with Jacobian matrix)
# print(x.grad)

# prevent graph calculation
# stop pytorch from calcualting gradient
# ex. during updating weights
# below are the methods
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

x = torch.randn(3, requires_grad=True)
print(x)
x.requires_grad_(False)
y = x.detach()
print(y)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    
    weights.grad.zero_() # clear previous gradient
    
# weights = torch.ones(4, requires_grad=True)

# # same in optimizer
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()


