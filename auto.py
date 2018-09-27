"""
The autoencoder.
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from copy import deepcopy
import math
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AUTO MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hidden', type=int, default=10, metavar='N', 
                        help='info. bottleneck dimensionality')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

class AUTO(nn.Module):
    def __init__(self):
        super(AUTO, self).__init__()

        self.fc1a = nn.Linear(784, 400)
        self.fc1b = nn.Linear(400, 200)
        self.fc2  = nn.Linear(200, args.hidden)
        self.fc3b = nn.Linear(args.hidden, 200)
        self.fc3a = nn.Linear(200, 400)
        self.fc4  = nn.Linear(400, 784)

    def encode(self, x):
        h1a = F.relu(self.fc1a(x))
        h1b = F.relu(self.fc1b(h1a))
        return self.fc2(h1b)

    def decode(self, z):
        h3b = F.relu(self.fc3b(z))
        h3a = F.relu(self.fc3a(h3b))
        return F.sigmoid(self.fc4(h3a))
   
    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z), z

if __name__ == "__main__":
    model = AUTO().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    return F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data.cuda())
        optimizer.zero_grad()
        recon_batch, z = model(data)
        torch.save(z, 'latent/'+ str(args.hidden) +  'D/train_e' + str(epoch) + '_b' + str(batch_idx))
        loss = loss_function(recon_batch, data)
        loss.backward()
#        print(loss.data)
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data.cuda())
        recon_batch, z = model(data)
        torch.save(z, 'latent/'+ str(args.hidden) + 'D/test_e' + str(epoch) + '_b' + str(i))
        test_loss += loss_function(recon_batch, data).data[0]
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        torch.save(model, 'autoencoders/epoch' + str(epoch))

