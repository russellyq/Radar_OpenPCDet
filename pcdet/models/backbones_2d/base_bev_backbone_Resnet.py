import numpy as np
import torch
import torch.nn as nn
import json
from .attention_layer import DeformableConv2d, SELayer, CBAM
from .inception_layer import ICConv2d


class BasicBlock_ICConv2d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, pattern=None, pattern_index=-1):
        super(BasicBlock_ICConv2d, self).__init__()
        pattern_index += 1

        self.conv1 = ICConv2d(pattern[pattern_index], inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        # pattern_index += 1

        self.conv2 = ICConv2d(pattern[pattern_index], planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01))
        else:
            self.downsample = lambda x: x
        
        self.attention = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_ICConv2d(nn.Module):
    expansion = 4
    # expansion = 1
    def __init__(self, inplanes, planes, stride=1, pattern=None, pattern_index=-1):
        super(Bottleneck_ICConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        pattern_index = pattern_index + 1
        self.conv2 = ICConv2d(pattern[pattern_index], planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01))
        else:
            self.downsample = lambda x: x
        self.attention = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)
        out += residual
        out = self.relu(out)
        return out


# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, inplanes, planes, stride=1, model_cfg=None):
#         super(BasicBlock, self).__init__()
        
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
#         if stride != 1 or inplanes != self.expansion*planes:
#             self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                                             nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01))
#         else:
#             self.downsample = lambda x: x
        
#         self.attention = nn.Sequential()
#         self.stride = stride

#     def forward(self, x):
#         residual = self.downsample(x)
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.attention(out)

#         out += residual
#         out = self.relu(out)
#         return out

class base_bev_backbone_Resnet_ICConv2d(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        pattern_path = '/home/yanqiao/Radar_OpenPCDet/tools/cfgs/VoD_models/ic_resnet50_k9.json'
        
        with open(pattern_path, 'r') as fin:
            self.pattern = json.load(fin)
            self.pattern_index = -1
            
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
        layer_nums = model_cfg.LAYER_NUMS
        layer_strides = model_cfg.LAYER_STRIDES
        num_filters = model_cfg.NUM_FILTERS
        self.inplane = 64
        if str(self.model_cfg.BLOCK) == 'BasicBlock_ICConv2d':
            block = BasicBlock_ICConv2d
        elif str(self.model_cfg.BLOCK) == 'Bottleneck_ICConv2d':
            block = Bottleneck_ICConv2d
        # block = Bottleneck_ICConv2d

        self.layer0 = self._make_layer(block, num_filters[0], layer_nums[0], 1)
        self.layer1 = self._make_layer(block, num_filters[1], layer_nums[1], layer_strides[0])
        self.layer2 = self._make_layer(block, num_filters[2], layer_nums[2], layer_strides[1])
        self.layer3 = self._make_layer(block, num_filters[3], layer_nums[3], layer_strides[2])
        
        self.delayer1 = self._make_delayer(num_filters[1] * block.expansion, 128, 1)
        self.delayer2 = self._make_delayer(num_filters[2] * block.expansion, 128, 2)
        self.delayer3 = self._make_delayer(num_filters[3] * block.expansion, 128, 4)
        
        assert len(self.pattern) == self.pattern_index + 1
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, self.pattern, self.pattern_index))
            self.inplane = planes * block.expansion
            self.pattern_index += 1
        return nn.Sequential(*layers)
    
    def _make_delayer(self, inplane, plane, upsample):
        layer = [nn.ConvTranspose2d(inplane, plane, upsample, stride=upsample, bias=False),
                 nn.BatchNorm2d(plane, eps=1e-3, momentum=0.01),
                 nn.ReLU()]
        return nn.Sequential(*layer)

    def _forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        dx1 = self.delayer1(x1)
        dx2 = self.delayer2(x2)
        dx3 = self.delayer3(x3)
        
        return torch.cat((dx1, dx2, dx3), dim=1)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = self._forward(spatial_features)
        data_dict['spatial_features_2d'] = x
        return data_dict
