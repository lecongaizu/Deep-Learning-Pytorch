import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'images'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--image_size', type=int, default=784, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--interval', type=int, default=2, help='interval betwen epoch')
parser.add_argument('--h_dim', type=int, default=400, help='h_dim')
parser.add_argument('--z_dim', type=int, default=20, help='z_dim')
opt = parser.parse_args()
print(opt)

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=opt.batch_size, 
                                          shuffle=True)


# VAE model
class VAE(nn.Module):
    def __init__(self, image_size = opt.image_size, h_dim=opt.h_dim, z_dim=opt.z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

avrage_recon_loss = []
avrage_kl_loss = []
step = []
# Start training
for epoch in range(opt.num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward pass
        x = x.to(device).view(-1, opt.image_size)
        x_reconst, mu, log_var = model(x)
        
        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, opt.num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
    avrage_recon_loss.append(reconst_loss.item())
    avrage_kl_loss.append(kl_div.item())
    step.append(epoch)

    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(opt.batch_size, opt.z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

# Plot loss network
plt.plot(step, avrage_recon_loss, label='Recontruct-loss')
plt.plot(step, avrage_kl_loss, label='KL-loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('loss.jpg')