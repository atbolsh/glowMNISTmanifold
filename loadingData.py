"""
Loading 
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
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


#Training epoch of autoencoder to load.
epoch = 64
dim   = 10

fileDir = 'latent/' + str(dim) + 'D' + '/'

fl     = os.listdir(fileDir)
trainL = [fileDir + x for x in fl if x.startswith('train_e' + str(epoch))]
testL  = [fileDir + x for x in fl if x.startswith( 'test_e' + str(epoch))]


class fList(torch.utils.data.Dataset):
    def __init__(self, fl):
        super(fList, self).__init__()
        self.fl = fl
    
    def __len__(self):
        return len(self.fl)
    
    def __getitem__(self, i):
        return torch.load(self.fl[i])

#batch_size is 1 because that's simpler. 
#It was already stored in unequal batches
trainD = torch.utils.data.DataLoader(fList(trainL), batch_size=1, shuffle=True) 
testD  = torch.utils.data.DataLoader(fList(testL),  batch_size=1, shuffle=True) 
      
if __name__ == '__main__':
    for i, data in enumerate(trainD):
        print(i)
        print(data.size())

    for i, data in enumerate(testD):
        print(i)
        print(data.size())
    


