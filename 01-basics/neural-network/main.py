import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms

# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
input_size = 784
hidden_size_1 = 500
hidden_size_2 = 256
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

class NeuralNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_size, hidden_size_1,hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1    = nn.Linear(input_size, hidden_size_1)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu   = nn.ReLU()
        self.fc3    = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)

# Loss function and optimizer 
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_item =  []
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size and input_size)
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # print(images)
        # Forward pass
        outputs = model(images)
        # print(outputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, i+1, total_step, loss.item()))
    loss_item.append(loss.item())

print(loss_item)

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