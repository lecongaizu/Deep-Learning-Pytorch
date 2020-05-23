import torch 

import torchvision 

import torch.nn as nn

import numpy as np

##---------------------
# Basic autograd example1
##----------------------

x = torch.rand(10,3)
y = torch.rand(10,2)

# Build full connected layer 
linear  = nn.Linear(3,2)

print('weight:', linear.weight)
print('bias:', linear.bias)

# Forward pass
pred = linear(x)

# Build loss function and optimizer 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)

#Comptue loss
loss = criterion(pred,y)
print('loss:', loss.item())

#Backward 1 time
loss.backward()

# Print out the gradients
print('dL/dW:', linear.weight.grad)
print('dL/db:', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

#Print out the loss after 1 step gradient descent 
pred = linear(x)

loss = criterion(pred,y)
print('loss after 1 step of gradient descent:', loss.item())

#Backward 2 times
loss.backward()

# Print out the gradients
print('dL/dW:', linear.weight.grad)
print('dL/db:', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

#Print out the loss after 1 step gradient descent 
pred = linear(x)

loss = criterion(pred,y)
print('loss after 2 step of gradient descent:', loss.item())
