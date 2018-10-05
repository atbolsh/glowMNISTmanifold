"""
Uses trained models to actually generate MNIST images.
"""

from model import *
from auto  import *

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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dim=12
#MOST IMPORTANT LINES.
#Look at "curve" to figure out the correct value here.
GLOWepoch = 10 
AUTOepoch = 12

glow = torch.load('glowVars/'    + str(dim) + 'D/epoch' + str(GLOWepoch))
auto = torch.load('autoencoders/'+ str(dim) +  'D/epoch' + str(AUTOepoch))

def newImages(good=True):
    for i in range(10):
        if good:
            latent = glow.sample(64)
            fname  = "sample_epoch" + str(GLOWepoch) + "_"
        else:
            target = torch.max(glow.sample(64))/3
            print(target)
            latent = torch.randn(64, dim).cuda()*target
            fname  = "control_"
        sample = auto.decode(latent).data.cpu()
        save_image(sample.view(64, 1, 28, 28),
               'samples/'    + str(dim) + 'D/' + fname + str(i) + '.png')
   
newImages()
newImages(good = False)
