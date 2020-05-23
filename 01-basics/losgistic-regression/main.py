import torch 

import torchvision 

import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt 

import torchvision.transforms as transforms

#Hyper parameters

input_size = 784

num_classes = 10

num_epoch = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST('../../data', train=False, transform=transforms.ToTensor())

# Data loader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Logistic regression 
model = nn.Linear(input_size, num_classes)

# Loss function and optimizer 
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size and input_size)
        images = images.reshape(-1, input_size)

        # print(images)
        # Forward pass
        outputs = model(images)
        # print(outputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # if (i+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, loss.item()))
# Test the model

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')