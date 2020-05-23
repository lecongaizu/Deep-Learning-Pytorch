import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms


# Device configuration 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
sequence_length =  28
input_size = 28
hidden_layer = 128 
num_layer = 2
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

class RNN(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_size, hidden_layer, num_layer, num_classes):
        super(RNN, self).__init__()
        self.hidden_layer =  hidden_layer
        self.num_layer =  num_layer
        self.lstm  =  nn.LSTM(input_size,hidden_layer, num_layer, batch_first=True)
        self.fc =  nn.Linear(hidden_layer, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 =  torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        c0 =  torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)

        # Forward propagate LSTM 
        out,_ = self.lstm(x, (h0,c0)) # out: tensor of shape (batch size, seq-lenght, hidden-layer)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, - 1, :])

        return out 

model = RNN(input_size, hidden_layer, num_layer, num_classes).to(device)


# Loss and optimizer 
# Loss fucntion and Optimizer 
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
avrage_loss = []
step = []

print("Starting for training model")

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}' 
                        .format(epoch+1, num_epoch,i+1, total_step, loss.item()))
    avrage_loss.append(loss.item())
    step.append(epoch)

# Plot loss network
plt.plot(step, avrage_loss, label='Network loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('loss.jpg')

# Save model
torch.save(model.state_dict(),'model.ckpt')

# Test model 
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