from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import pdb



class Di(nn.Module):

    def __init__(self,num_features=0,norm=False,num_classes=0, dropout=0,radius=1., thresh=0.5):
        super(Di, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #self.classifier = nn.Linear(2048, num_classes)
        self.num_features=num_features # feature dimension
        self.num_classes=num_classes
        self.norm=norm

        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes,bias=True)
            init.normal(self.classifier.weight, std=0.001)
            init.constant(self.classifier.bias, 0)
#*****************************************************
    def forward(self, x):

        x = self.base(x)
        if self.norm:
            x = self.bn(x)
        if self.num_classes > 0:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        return x
	




