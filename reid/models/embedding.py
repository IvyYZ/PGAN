import math
import copy
from torch import nn
import torch
import torch.nn.functional as F
import pdb
from torch.nn import init

class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        #pdb.set_trace()
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)

        return x
    

class Di(nn.Module):

    def __init__(self,num_features=0,norm=False,num_classes=0, dropout=0,radius=1., thresh=0.5):
        super(Di, self).__init__()
#        resnet50 = torchvision.models.resnet50(pretrained=True)
#        self.base = nn.Sequential(*list(resnet50.children())[:-2])
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

        #x = self.base(x)
        #x=x.unsqueeze(1)
        x = F.avg_pool2d(x, x.size()[2:])
        if self.norm:
            x = self.bn(x)
        if self.num_classes > 0:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        return x

