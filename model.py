"""
GLOW model, to create the latent space distribution.
"""

from LU  import *
from affine import *
from actnorm import *

from sklearn.datasets import load_boston

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
os.environ["CUDA_VISIBLE_DEVICES"]="5"

dim=5

#Now we have DataLoaders trainD and testD
import loadingData as lD
if __name__ == "__main__":
    trainD, testD = lD.main(dim=dim, epoch=10)
#Build the network

class flowGAN(nn.Module):
    def __init__(self, n, k = 10, backend=64):
        """Even n preferred"""
        self.n = n
        self.k = k
        super(flowGAN, self).__init__()
        self.lin  = nn.ModuleList([LU(k) for i in range(n+1)])
        self.norm = nn.ModuleList([ActNorm(k) for i in range(n)])
        aff = []
        for i in range(n):
            if i % 2 == 0:
                aff.append(affineDiscrete(k, hidden=backend))
            else:
                aff.append(affineDiscrete(k, hidden=backend))
#                aff.append(backLayer(2))
        self.aff = nn.ModuleList(aff)
        self.n = n
 
    def forward(self, x):
        y, lJ = self.lin[0](x)
        for i in range(self.n):
            y, nlJ = self.norm[i](y)
            lJ = nlJ + lJ
            y, nlJ = self.aff[i](y)
            lJ = nlJ + lJ
            y, nlJ = self.lin[i+1](y)
            lJ = lJ + nlJ
        return y, math.log(1/(2*math.pi)) - torch.sum(y*y, 1)/2 + lJ

    def sample(self, n, x = None):
        if (type(x) == type(None)):
            x = Variable(torch.randn(n, self.k)).cuda()
        x, _ = self.lin[-1].pushback(x)
        for i in range(1, self.n+1):
            x, _ = self.aff[-i].pushback(x)
            x, _ = self.norm[-i].pushback(x)
            x, _ = self.lin[-1-i].pushback(x)
        return x



if __name__ == "__main__":
    batchsize = 1000
    epochnum = 200000


#alpha=10
if __name__ == "__maine__":
    model = flowGAN(5, k=dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

if __name__ == "__main__":
    trainingCurve = open('glowVars/' + str(model.k) + 'D/curve', 'w')
    trainingCurve.write('Epoch\t\tLoss\n')
    trainingCurve.close()  
    for epoch in range(epochnum):
        
        for i, data in enumerate(trainD):
            batchSize = data.size(1)
            xu = Variable(data.view(batchSize, model.k)).cuda()
            print(xu.size())   
            model.train()
            model.zero_grad()    
            yu, lu = model(xu)
            with torch.no_grad():
                xd = model.sample(batchsize)

#            print(yu[:10])
#            print((math.log(2*math.pi) + lu + 0.5*torch.sum(yu*yu, 1))[:10])
            yd, ld = model(xd.detach())
            pu = torch.exp(lu)
#        pd = torch.exp(ld)
#        print(pd)
        
#        loss = L(yu, target)#/batchsize # + torch.sum(ld)
            loss = 0 - torch.sum(lu) + torch.sum(ld)

            loss.backward()
            
            optimizer.step()
        
        model.eval()
        for i, data in enumerate(testD):
            batchSize = data.size(1)
            xu = Variable(data.view(batchSize, model.k)).cuda()      
            yu, lu = model(xu)
#        print('adversarial dif = ' + str(torch.sum(ld) - torch.sum(lu)))
       
        if epoch%1 == 0:
            torch.save(model, 'glowVars/' + str(model.k) + 'D/epoch' + str(epoch))
            trainingCurve = open('glowVars/' + str(model.k) + 'D/curve', 'a')
            trainingCurve.write(str(epoch) + '\t\t' + str(loss) + '\n')
            trainingCurve.close()








