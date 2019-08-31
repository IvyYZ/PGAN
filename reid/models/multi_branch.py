from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class SiameseNet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(SiameseNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1, x2):
        #pdb.set_trace()
        x1, x2 = self.base_model(x1), self.base_model(x2)
        
        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1= x1.view(x1.size(0), -1)
        x2 = F.avg_pool2d(x2, x2.size()[2:])
        x2= x2.view(x2.size(0), -1)

        if self.embed_model is None:
            return x1, x2
        return x1, x2, self.embed_model(x1, x2)

class ENet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(ENet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1):
        #pdb.set_trace()
        x1= self.base_model(x1)
        
        if self.embed_model is None:
            return x1
        x2,x3,x4,x5=self.embed_model(x1)
        return x2,x3,x4,x5

class DNet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(DNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1):
        #pdb.set_trace()
        x1= self.base_model(x1)
        
        if self.embed_model is None:
            return x1
        
        return x1,self.embed_model(x1)
