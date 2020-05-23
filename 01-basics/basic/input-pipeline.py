import torch 

import torchvision 

import torch.nn as nn

import torchvision.transforms as transforms

import numpy as np

import torch.utils.data

# Download and contruct CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, transform=transforms.ToTensor(), download=True)

# Fetch one data pair (read data from disk)
image, label = train_dataset[0]
print(image.size())
print(label)

# # Data loader (this provides queues and threads in a very simple way )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

# # When iteration starts, queue and thread start to load data from files
data_iter = iter(train_loader)

# # Mini-batch images and label
images, labels = data_iter.next()

for images, labels in train_loader:
    # Training code should be write here 
    pass

# You should build your customer dataset as below
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1. Initialize file paths or a list of file names
        pass
    def __getitem__(self, index):
        # todo
        # 1. READ ONE DATA FROM FILE (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess data (torchvision.transform)
        # 3. Return data pair (image and label)
        pass
    def __len__(self):
        # Shoud change 0 to the total size of your dataset
        pass

custom_dataset = CustomDataset()

train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)

# Save and load the entire model
torch.save(model, 'model.ckpt')
model = torch.load('model.ckpt')