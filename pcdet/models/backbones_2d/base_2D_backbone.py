import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.relu2(out)
        return out

class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        inplane = 3
        outplane = [16, 32, 64, 64]
        
        self.conv1 = BasicBlock(inplane, outplane[0], stride=1)
        self.conv2 = BasicBlock(outplane[0], outplane[1], stride=2)
        self.conv3 = BasicBlock(outplane[1], outplane[2], stride=2)
        self.conv4 = BasicBlock(outplane[2], outplane[3], stride=2)
    
    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        h, w = x_in.shape[2:]
        x1 = nn.functional.interpolate(x1, (h, w), mode='bilinear')
        x2 = nn.functional.interpolate(x2, (h, w), mode='bilinear')
        x3 = nn.functional.interpolate(x3, (h, w), mode='bilinear')
        x4 = nn.functional.interpolate(x4, (h, w), mode='bilinear')
        
        return x1,x2, x3, x4