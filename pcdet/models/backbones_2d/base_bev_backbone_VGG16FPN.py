import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import SELayer

class BaseBEVBackbone_Resnet18(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        # if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
        #     assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
        #     num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        #     upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        # else:
        #     upsample_strides = num_upsample_filters = []
        # c_in = sum(num_upsample_filters)
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

        self.net = FPN(BasicBlock, [2,2,2,2], self.model_cfg)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = self.net(spatial_features)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BaseBEVBackbone_Resnet34(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.net = FPN(BasicBlock, [3,4,6,3], self.model_cfg)
    
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.net(spatial_features)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BaseBEVBackbone_Resnet50(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.net = FPN(Bottleneck, [3,4,6,3], self.model_cfg)
    
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.net(spatial_features)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BaseBEVBackbone_Resnet101(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.net = FPN(Bottleneck, [3,4,6,3], self.model_cfg)
    
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.net(spatial_features)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, model_cfg=None):
        super(BasicBlock, self).__init__()
        self.model_cfg = model_cfg
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.se = SELayer(self.expansion*planes) if self.model_cfg.SE_ATTENTION == True else nn.Sequential()
        
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
            ) if str(self.model_cfg.RESNET) == 'A' else  nn.Sequential(
                                                    nn.AvgPool2d(kernel_size=2, stride=stride),
                                                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                                                    nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
                                                )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, model_cfg=None):
        super(Bottleneck, self).__init__()
        self.model_cfg = model_cfg
 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.se = SELayer(self.expansion*planes) if self.model_cfg.SE_ATTENTION == True else nn.Sequential()
        
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
            )
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.relu3(out)
        return out

# class FPN(nn.Module):
#     def __init__(self, block, num_blocks, model_cfg, num_filters = [64, 128, 256, 512]) -> None:
#         super().__init__()
        
#         self.model_cfg = model_cfg
#         self.in_planes = self.model_cfg.IN_PLANES # 64
#         self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 384
        
        
        
#         # Bottom-up layers
#         self.layer1 = self._make_layer(block, num_filters[0], num_blocks[0], stride=1)
        
#         self.layer2 = self._make_layer(block, num_filters[1], num_blocks[1], stride=2)
        
#         self.layer3 = self._make_layer(block, num_filters[2], num_blocks[2], stride=2)
        
#         self.layer4 = self._make_layer(block, num_filters[3], num_blocks[3], stride=2)
        
#         # Top layer
#         self.toplayer = nn.Conv2d(block.expansion * num_filters[3], self.num_bev_features, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
#         # Smooth layers
#         self.smooth2 = nn.Conv2d(self.num_bev_features, self.num_bev_features, kernel_size=3, stride=1, padding=1)
        
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(block.expansion * num_filters[2], self.num_bev_features, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d( block.expansion * num_filters[1], self.num_bev_features, kernel_size=1, stride=1, padding=0)
        
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         # layers = [nn.ZeroPad2d(1)]
#         layers = []
        
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self.model_cfg))
#             self.in_planes = planes * block.expansion
    
#         return nn.Sequential(*layers)

#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         # return F.interpolate(x, size=(H,W), mode=str(self.model_cfg.MODE)) + y
#         return F.interpolate(x, size=(H,W)) + y

#     def forward(self, x):
#         # print('\nx shape:', x.shape)
#         # Bottom-up
#         c2 = self.layer1(x) # self.expansion * num_filters[0], H, W
#         # print('\nc2 shape: ', c2.shape)

        
#         c3 = self.layer2(c2) # self.expansion * num_filters[1], H/2, W/2
#         # print('\nc3 shape: ', c3.shape)
        
#         c4 = self.layer3(c3) # self.expansion * num_filters[2], H/4, W/4
#         # print('\nc4 shape: ', c4.shape)

#         c5 = self.layer4(c4) # self.expansion * num_filters[3], H/8, W/8
#         # print('\nc5 shape: ', c5.shape)
#         # Top-down
#         p5 = self.toplayer(c5)  # p5: self.num_bev_features, H/8， W/8
#         # print('\np5 shape: ', p5.shape)
#         p4 = self._upsample_add(p5, self.latlayer1(c4)) # self.num_bev_features, H/4, W/4
#         # print('\np4 shape: ', p4.shape)
#         p3 = self._upsample_add(p4, self.latlayer2(c3)) # self.num_bev_features, H/2, W/2
#         # print('\np3 shape: ', p3.shape)
     
#         # Smooth
#         p3 = self.smooth2(p3)
#         # print('\np3 shape: ', p3.shape)

#         return p3



    



class FPN(nn.Module):
    def __init__(self, block, num_blocks, model_cfg, num_filters = [64, 128, 256, 512]) -> None:
        super().__init__()
        
        self.model_cfg = model_cfg
        self.in_planes = self.model_cfg.IN_PLANES # 64
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 384
        
        
        
        # Bottom-up layers
        self.layer1 = self._make_layer(block, num_filters[0], num_blocks[0], stride=1)
        
        self.layer2 = self._make_layer(block, num_filters[1], num_blocks[1], stride=2)
        
        self.layer3 = self._make_layer(block, num_filters[2], num_blocks[2], stride=2)
        
        self.layer4 = self._make_layer(block, num_filters[3], num_blocks[3], stride=2)
        
        # Top layer
        self.toplayer = nn.Conv2d(block.expansion * num_filters[3], self.num_bev_features, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
        # Smooth layers
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(block.expansion * num_filters[2], self.num_bev_features, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( block.expansion * num_filters[1], self.num_bev_features, kernel_size=1, stride=1, padding=0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # layers = [nn.ZeroPad2d(1)]
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.model_cfg))
            self.in_planes = planes * block.expansion
    
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        # return F.interpolate(x, size=(H,W), mode=str(self.model_cfg.MODE)) + y
        return F.interpolate(x, size=(H,W)) + y

    def forward(self, x):
        # print('\nx shape:', x.shape)
        # Bottom-up
        c2 = self.layer1(x) # self.expansion * num_filters[0], H, W
        # print('\nc2 shape: ', c2.shape)

        
        c3 = self.layer2(c2) # self.expansion * num_filters[1], H/2, W/2
        # print('\nc3 shape: ', c3.shape)
        
        c4 = self.layer3(c3) # self.expansion * num_filters[2], H/4, W/4
        # print('\nc4 shape: ', c4.shape)

        c5 = self.layer4(c4) # self.expansion * num_filters[3], H/8, W/8
        # print('\nc5 shape: ', c5.shape)
        # Top-down
        p5 = self.toplayer(c5)  # p5: self.num_bev_features, H/8， W/8
        # print('\np5 shape: ', p5.shape)
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # self.num_bev_features, H/4, W/4
        # print('\np4 shape: ', p4.shape)
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # self.num_bev_features, H/2, W/2
        # print('\np3 shape: ', p3.shape)
     
        # Smooth

        return p3, p4, p5



    
