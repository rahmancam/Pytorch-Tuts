# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    }

# Import data
data_directory = 'datasets/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_directory, x), data_transforms[x]) for x in sets
    }

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in sets}
dataset_sizes = { x: len(image_datasets[x]) for x in sets }
class_names = image_datasets['train'].classes
print(f'Class names : {class_names}')

# Train model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)
        
        # Each epoch has a training and validation phase
        for phase in sets:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model. eval() # set model to evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for images, labels in data_loaders[phase]:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forwards pass
                # Track history only if in training mode
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize only if in training mode
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step() 
                
                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
            
            # deep copy the model
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
            
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best validation accuracy: {best_accuracy:.4f}')
            
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model

# Import pre-trained model
# Option 1 - update gradients to all layers
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features # Last layer number of input features
model.fc = nn.Linear(num_features, 2) # New layer for our classification
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion,  optimizer, step_lr_scheduler, num_epochs=20)


# Option 2 - Freeze all the layers except last one (disable gradients for all those layers)
model = models.resnet18(pretrained=True)

# disable gradients
for param in model.parameters():
    param.requires_grad = False
    
num_features = model.fc.in_features # Last layer number of input features
model.fc = nn.Linear(num_features, 2) # New layer for our classification
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion,  optimizer, step_lr_scheduler, num_epochs=20)

    
