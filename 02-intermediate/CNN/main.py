import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms


# Device configuration 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_classes = 10
num_epoch = 10
batch_size = 100
learning_rate = 0.0001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST('../../data', train=False, transform=transforms.ToTensor())

# Data loader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Convnet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, num_classes):
        super(Convnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = Convnet(num_classes).to(device)

# Loss fucntion and Optimizer 
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
avrage_loss = []
step = []
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
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
    print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, loss.item()))
    avrage_loss.append(loss.item())
    step.append(epoch)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
# Plot loss network
plt.plot(step, avrage_loss, label='Network loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('loss.jpg')

# Save model
torch.save(model.state_dict(),'model.ckpt')
model.eval() # eval mode (batchnorm uses moving mean and variance insted of mini bnatch mean and variance)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
