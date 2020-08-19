# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# device config
device = torch.device('cpu')
print(f'Avialble Device : {device}')
 
# hyper parameters
input_size = 784 # 28 * 28
num_classes = 10
hidden_size = 100
num_epochs = 2
batch_size = 100
lr = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr =lr)


writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        
        # forward
        outputs = model(images)
        
        #loss
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()
        
        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.3f}')
        
with torch.no_grad():
    n_samples = 0
    n_correct = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    accuracy = 100 * n_correct / n_samples
    print(f'accuracy: {accuracy:.3f}')
        
    
        
    