"""
This is the module for the RealNVP-style affine layer, for a 2D input variable.
"""

#import nln

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



class transformerFC(nn.Module):
    def __init__(self, start, end, k=2, hidden=64, mult=True):
        super(transformerFC, self).__init__()
        self.k = k
        self.mult = mult

        self.b_up = nn.Linear(start, hidden)
        self.b_fc = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(k)])
        self.b_down = nn.Linear(hidden, end) 
        self.b_down.weight.data *= math.sqrt(float(end)/hidden)
        self.b_down.bias.data *= math.sqrt(float(end)/hidden)
               
        if mult:
            self.m_up = nn.Linear(start, hidden)
            self.m_fc = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(k)])
            self.m_down = nn.Linear(hidden, end) 
            self.m_down.weight.data *= math.sqrt(float(end)/hidden)
            self.m_down.bias.data *= math.sqrt(float(end)/hidden) 
       
    def m_and_b(self, base):
        h1b = F.relu(self.b_up(base))
        for i in range(self.k):
            h1b = F.relu(self.b_fc[i](h1b))
        b = self.b_down(h1b)
          
        if self.mult: 
            h1m = F.relu(self.m_up(base))
            for i in range(self.k):
                h1m = F.relu(self.m_fc[i](h1m))
            m = self.m_down(h1m) 
           
            return m, b
        else:
            return b

    def forward(self, x, base):
        if self.mult:
            m, b = self.m_and_b(base)
            y = torch.exp(m)*x + b
            return y, torch.sum(m, dim=1)

        else:
            b = self.m_and_b(base)
            y = x + b
            return y, 0.
                       
    def pushback(self, y, base):
        if self.mult:
            m, b = self.m_and_b(base)
            x = (y - b)/torch.exp(m)
            return x, torch.sum(m, dim=1)
        else:
            b = self.m_and_b(base)
            x = y - b
            return x, 0.   

#Glow-style
class affineHalves(nn.Module):
    def __init__(self, n, k = 2, hidden=64):
        super(affineHalves, self).__init__()
        self.n = n
        self.h = int(n/2)
        
        self.t1 = transformerFC(self.h, self.n - self.h, k, hidden, True)
        self.t2 = transformerFC(self.n - self.h, self.h, k, hidden, False)
    
    def forward(self, x):
        s = x.size(0)
        y1, l1 = self.t1(x[:, self.h:].view(s, self.n - self.h), x[:, :self.h].view(s, self.h))
        y0, l0 = self.t2(x[:, :self.h].view(s, self.h), y1)
        return torch.cat((y0, y1), 1), l0 + l1
        
    def pushback(self, y):
        s = y.size(0)
        x0, l0 = self.t2.pushback(y[:, :self.h].view(s, self.h), y[:, self.h:].view(s, self.n - self.h))
        x1, l1 = self.t1.pushback(y[:, self.h:].view(s, self.n - self.h), x0)
        return torch.cat((x0, x1), 1), l0 + l1
        
#Full Jacobian
class affineDiscrete(nn.Module):      
    def __init__(self, n, k = 2, hidden=64):
        super(affineDiscrete, self).__init__()
        self.n = n
        
        self.t1 = nn.ModuleList([transformerFC(i,         1, k, hidden, True)  for i in range(1, self.n)])
        self.t2 = nn.ModuleList([transformerFC(self.n -i, 1, k, hidden, False) for i in range(1, self.n)])
    
    def forward(self, x):
        s = x.size(0)
        #This could be parallelized . . . not sure how. Look up special torch structures.
        ys = [self.t1[i-1](x[:, i].view(s, 1), x[:, :i].view(s, i)) for i in range(1, self.n)]
        l = sum([q[1] for q in ys])
        y = torch.cat(tuple([x[:, 0].view(s, 1)] + [q[0] for q in ys]), 1)
        print(y[0])
        for i in range(self.n-1, 0, -1):
            print(y[0, -1-i].view(1, 1))

        zs = [self.t2[-i](y[:, -1-i].view(s, 1), y[:, -i:].view(s, i)) for i in range(self.n - 1, 0, -1)]
        j = sum([q[1] for q in zs])
        z = torch.cat(tuple([q[0] for q in zs] + [y[:, -1].view(s, 1)]), 1)

        return z, l+j
        
    def pushback(self, z):
        s = z.size(0)

        l = 0.
        y = z[:, -1].view(s, 1) + 0
#        print(y.size())

        for i in range(1, self.n):
            ny, nl = self.t2[-i].pushback(z[:, -1-i].view(s, 1), y)
            l = nl + l
            y = torch.cat((ny, y), 1)
#            print(y.size())

        x = y[:, 0].view(s, 1) 
        for i in range(1, self.n):
            nx, nl = self.t1[i-1].pushback(y[:, i].view(s, 1), x)
            l = nl + l
            x = torch.cat((x, nx), 1)
        
        return x, l

class affineSegs(nn.Module):      
    def __init__(self, n, maxSeg=32, k = 2, hidden=64):
        super(affineDiscrete, self).__init__()
        self.n = n
        self.seg = min(n, maxSeg)

        self.step  = int(n/(self.seg))
        self.extra = self.n - self.step*self.seg
        
        #Sets of dimensions
        self.sizes = [self.step+1 for k in range(self.extra)] + [self.step for k in range(self.extra, self.seg)]
        self.t1_bases = [0] + [torch.sum(self.sizes[:i]) for i in range(1, self.seg)]
        self.t2_bases = [torch.sum(self.sizes[i+1:]) for i in range(0, self.seg - 1)] + [0]
        
        self.t1 = nn.ModuleList([transformerFC(self.t1_bases[i], self.sizes[i], k, hidden, True)  for i in range(1, self.seg)])
        self.t2 = nn.ModuleList([transformerFC(self.t2_bases[i], self.sizes[i], k, hidden, False) for i in range(0, self.seg-1)])

    def _t1_forward(self, x, i):
        """Forward step for the ith segment, on t1""" 
        s = x.size(0)
        
        #Slice of dimensions to be pushed through
        target = x[:, self.t1_bases[i].view(s, self.sizes[i])
        
        #Base for the transformer
        base   = x[:, :sum(self.sizes[:i])].view(s, sum(self.sizes[:i]))
        
        return self.t1[i - 1](target, base)
    
    def forward(self, x):
        s = x.size(0)
        #This could be parallelized . . . not sure how. Look up special torch structures.
        ys = [self.t1[i-1](x[:, i].view(s, 1), x[:, :i].view(s, i)) for i in range(1, self.n)]
        l = sum([q[1] for q in ys])
        y = torch.cat(tuple([x[:, 0].view(s, 1)] + [q[0] for q in ys]), 1)
        print(y[0])
        for i in range(self.n-1, 0, -1):
            print(y[0, -1-i].view(1, 1))

        zs = [self.t2[-i](y[:, -1-i].view(s, 1), y[:, -i:].view(s, i)) for i in range(self.n - 1, 0, -1)]
        j = sum([q[1] for q in zs])
        z = torch.cat(tuple([q[0] for q in zs] + [y[:, -1].view(s, 1)]), 1)

        return z, l+j
        
    def pushback(self, z):
        s = z.size(0)

        l = 0.
        y = z[:, -1].view(s, 1) + 0
#        print(y.size())

        for i in range(1, self.n):
            ny, nl = self.t2[-i].pushback(z[:, -1-i].view(s, 1), y)
            l = nl + l
            y = torch.cat((ny, y), 1)
#            print(y.size())

        x = y[:, 0].view(s, 1) 
        for i in range(1, self.n):
            nx, nl = self.t1[i-1].pushback(y[:, i].view(s, 1), x)
            l = nl + l
            x = torch.cat((x, nx), 1)
        
        return x, l


#Intermediate, flexible compromise
class affineSteps(nn.Module):      
    def __init__(self, n, maxSeg = 32, k = 2, hidden=64):
        super(affineSteps, self).__init__()
        self.n   = n
        self.seg = min(n, maxSeg)

        self.step  = int(n/(self.seg))
        self.extra = self.n - self.step*self.seg
        
        #Sets of dimensions
        self.sizes = [self.step+1 for k in range(self.extra)] + [self.step for k in range(self.extra, self.seg)]
        
        self.t1 = nn.ModuleList([transformerFC(sum(self.sizes[:i]),  self.sizes[i],    k,  hidden, True)  for i in range(1, self.seg)])
        self.t2 = nn.ModuleList([transformerFC(sum(self.sizes[-i:]), self.sizes[-1-i], k,  hidden, False) for i in range(self.seg -1, 0, -1)])
    
    def _t1_forward(self, x, i):
        """Forward step for the ith segment, on t1""" 
        s = x.size(0)
        
        #Slice of dimensions to be pushed through
        target = x[:, sum(self.sizes[:i]):sum(self.sizes[:i+1])].view(s, self.sizes[i])
        
        #Base for the transformer
        base   = x[:, :sum(self.sizes[:i])].view(s, sum(self.sizes[:i]))
        
        return self.t1[i - 1](target, base)
 
    def _t2_forward(self, x, i):
        """Forward step for the ith segment, on t2""" 
        s = x.size(0)
        
        #Slice of dimensions to be pushed through
        #Not most efficient code, but seg is assumed to be on the order of 10^2 max
        target = x[:, -sum(self.sizes[-1-i:]):-sum(self.sizes[-i:])].view(s, self.sizes[-1-i])
        print(target[0])
         
        #Base for the transformer
        base   = x[:, -sum(self.sizes[-i:]):].view(s, sum(self.sizes[-i:]))
        
        return self.t2[-i](target, base)

    def forward(self, x):
        s = x.size(0)
        #This could be parallelized . . . not sure how. Look up special torch structures.
        ys = [self._t1_forward(x, i) for i in range(1, self.seg)]
        l = sum([q[1] for q in ys])
        y = torch.cat(tuple([x[:, :self.sizes[0]].view(s, self.sizes[0])] + [q[0] for q in ys]), 1)
#        print(y.size())
        print(y[0])

        zs = [self._t2_forward(x, i) for i in range(self.seg - 1, 0, -1)]
        j = sum([q[1] for q in zs])
        z = torch.cat(tuple([q[0] for q in zs] + [y[:, -self.sizes[-1]:].view(s, self.sizes[-1])]), 1)

        return z, l+j
        
    def pushback(self, z):
        s = z.size(0)

        l = 0.
        y = z[:, -self.sizes[-1]:].view(s, self.sizes[-1]) + 0
#        print(y.size())

        for i in range(1, self.seg):
            target = z[:, -sum(self.sizes[-1-i:]):-sum(self.sizes[-i:])].view(s, self.sizes[-1-i])
            ny, nl = self.t2[-i].pushback(target, y)
            l = nl + l
            y = torch.cat((ny, y), 1)
#            print(y.size())

        x = y[:, :self.sizes[0]].view(s, self.sizes[0]) 
        for i in range(1, self.seg):
            target = y[:, sum(self.sizes[:i]):sum(self.sizes[:i+1])].view(s, self.sizes[i])
            nx, nl = self.t1[i-1].pushback(target, x)
            l = nl + l
            x = torch.cat((x, nx), 1)
        
        return x, l

  
