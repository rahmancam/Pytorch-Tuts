# -*- coding: utf-8 -*-
import torch
import numpy as np

# Scalar
x = torch.empty(1)
print(x)

# 1-D vectors
x = torch.empty(3)
print(x) 

# 2-D
x = torch.empty(2, 2)
print(x)

# 3-D
x = torch.empty(2, 2, 3)
print(x)

# 4-D
x = torch.empty(2, 2 , 3, 4)
print(x)

# Tensor with random values
x = torch.rand(2, 2)
print(x)

# with zeros
x = torch.zeros(2, 2)
print(x)

# with ones
x = torch.ones(2, 2)
print(x)

# with data type
x = torch.ones(2, 2, dtype=torch.int)
print(x.dtype)
print(x.size())

x = torch.ones(2, 2, dtype=torch.float16)
print(x.dtype)
print(x.size())

# tensor
x = torch.tensor([2.5, 1.0])
print(x)


# Addition
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
print(z)
# same operation
z = torch.add(x, y)
print(z)

# in place addition (mutation)
y.add_(x)
print(y)

# subtraction
z = x - y
z = torch.sub(x, y)
print(z)

# element wise multiplication
z = x * y
z = torch.mul(x, y)
print(z)

# element wise divison
z = x / y
z = torch.div(x, y)
print(z)

# slicing
x = torch.rand(5, 3)
print(x)
print(x[1, :]) # print second row and all columns
print(x[1, 1]) # print one element
print(x[1, 1].item()) # value

# reshaping (resize tensors)
x = torch.rand(4, 4)
print(x)
y = x.view(16) # 1-D vector
print(y)
y = x.view(-1, 8) # pytorch automatically calculates other dim
print(y)
print(y.size())

# tensors to numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1) # Both a and b points to same memory location (cpu)
print(a)
print(b)

# numpy array to tensors
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b) # tensor got modified (cpu)


# GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y # performed on GPU
    # z.numpy() # returns error, can handle only cpu
    z = z.to('cpu')

# Tell pytoch to calculate gradient in optimization step
x = torch.ones(5, requires_grad=True) # by default false
print(x)

    

