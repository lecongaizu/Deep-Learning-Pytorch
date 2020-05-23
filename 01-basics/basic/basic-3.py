import torch 

import torchvision 

import torch.nn as nn

import numpy as np

# Create a numpy array 
x = np.array([[1,2],[3,4]])
print(x)
# Convert numpy to tensor 
y = torch.from_numpy(x)
print(y)
# Convert the torch tensor to a numpy array 
z = y.numpy()
print(z)