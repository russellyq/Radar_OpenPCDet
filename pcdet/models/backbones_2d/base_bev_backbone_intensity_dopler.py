from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from .attention_layer import DeformableConv2d, SELayer, CBAM

class BaseBEVBackboneIntensityDoppler(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
        block = BasicBlockGeo 
        self.net = BackbonewithResnet(block, model_cfg)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        x = self.net(data_dict)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BackbonewithResnet(nn.Module):
    def __init__(self, block, model_cfg) -> None:
        super().__init__()
        
        add_planes = 3
        self.model_cfg = model_cfg
               
        num_filters = [128, 256, 512]
        # num_filters = [64, 128, 256]
        layer_nums = [3, 5, 5]
        
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(num_filters[0] + add_planes, num_filters[0], kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_filters[0], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[0], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[0], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[0], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(num_filters[0] + add_planes, num_filters[1], kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU(),
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU()
        )                
        self.block3 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(num_filters[1]  + add_planes, num_filters[2], kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU(),
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU(),
            nn.Conv2d(num_filters[2], num_filters[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_filters[2], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn. ReLU()
        )
        self.deblock1 = self._make_delayer(num_filters[0], 1, 1)
        self.deblock2 = self._make_delayer(num_filters[1], 2, 2)
        self.deblock3 = self._make_delayer(num_filters[2], 4, 4)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    
    def _make_delayer(self, plane, kernel, stride):
        delayers = [nn.ConvTranspose2d(plane, 128, kernel, stride=stride, bias=False),
                    nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                    nn.ReLU()]
        return nn.Sequential(*delayers)

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        add_features = data_dict['add_features']
        # add_features = None
        x = torch.cat([add_features, spatial_features], 1)
        # x = spatial_features
        b1 = self.block1(x)
        
        add_features = self.maxpool(add_features)
        x = torch.cat([b1, add_features], 1)
        # x = b1
        b2 = self.block2(x)
        
        add_features = self.maxpool(add_features)
        x = torch.cat([b2, add_features], 1)
        # x = b2
        b3 = self.block3(x)

        d1 = self.deblock1(b1)
        d2 = self.deblock2(b2)
        d3 = self.deblock3(b3)
        return torch.cat([d1, d2, d3], dim=1)

# class BackbonewithResnet(nn.Module):
#     def __init__(self, block, model_cfg) -> None:
#         super().__init__()
        
#         add_planes = 0
#         self.model_cfg = model_cfg
        
#         # num_filters = [64, 128, 256]
        
#         num_filters = [128, 256, 512]
#         layer_nums = [3, 5, 5]
        
#         self.block1_layer1 = block(in_planes=num_filters[0], planes=num_filters[0], stride=2, model_cfg=model_cfg, add_planes=add_planes)
#         self.block1_layer2 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         # self.block1_layer3 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
#         self.block2_layer1 = block(in_planes=num_filters[0], planes=num_filters[1], stride=2, model_cfg=model_cfg, add_planes=add_planes)
#         self.block2_layer2 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         self.block2_layer3 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         # self.block2_layer4 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         # self.block2_layer5 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
#         self.block3_layer1 = block(in_planes=num_filters[1], planes=num_filters[2], stride=2, model_cfg=model_cfg, add_planes=add_planes)
#         self.block3_layer2 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         self.block3_layer3 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         # self.block3_layer4 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
#         # self.block3_layer5 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
#         self.deblock1 = self._make_delayer(num_filters[0], 1, 1)
#         self.deblock2 = self._make_delayer(num_filters[1], 2, 2)
#         self.deblock3 = self._make_delayer(num_filters[2], 4, 4)

    
#     def _make_delayer(self, plane, kernel, stride):
#         delayers = [nn.ConvTranspose2d(plane, 128, kernel, stride=stride, bias=False),
#                     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
#                     nn.ReLU()]
#         return nn.Sequential(*delayers)

#     def forward(self, data_dict):
#         spatial_features = data_dict['spatial_features']
#         # add_features = data_dict['add_features']
#         add_features = None
        
#         b11, add_features = self.block1_layer1(spatial_features, add_features)
#         b12, add_features = self.block1_layer2(b11, add_features)
#         # b13, add_features = self.block1_layer3(b12, add_features)
        
#         # b21, add_features = self.block2_layer1(b13, add_features)
#         b21, add_features = self.block2_layer1(b12, add_features)
#         b22, add_features = self.block2_layer2(b21, add_features)
#         b23, add_features = self.block2_layer3(b22, add_features)
#         # b24, add_features = self.block2_layer4(b23, add_features)
#         # b25, add_features = self.block2_layer5(b24, add_features)
        
#         # b31, add_features = self.block3_layer1(b25, add_features)
#         b31, add_features = self.block3_layer1(b23, add_features)
#         b32, add_features = self.block3_layer2(b31, add_features)
#         b33, add_features = self.block3_layer3(b32, add_features)
#         # b34, add_features = self.block3_layer4(b33, add_features)
#         # b35, add_features = self.block3_layer5(b34, add_features)
        
#         # d1 = self.deblock1(b13)
#         # d2 = self.deblock2(b25)
#         # d3 = self.deblock3(b35)
#         d1 = self.deblock1(b12)
#         d2 = self.deblock2(b23)
#         d3 = self.deblock3(b33)
#         return torch.cat([d1, d2, d3], dim=1)
        
class BaseBEVBackboneIntensityDopplerResnet18(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
        block = BasicBlockGeo 
        self.net = BackbonewithResnet18(block, model_cfg)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        x = self.net(data_dict)
        data_dict['spatial_features_2d'] = x
        return data_dict

class BackbonewithResnet18(nn.Module):
    def __init__(self, block, model_cfg) -> None:
        super().__init__()
        add_planes = 0
        self.model_cfg = model_cfg
        num_filters = [64, 64, 128, 256]
        self.block1_layer1 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block1_layer2 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block2_layer1 = block(in_planes=num_filters[0], planes=num_filters[1], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block2_layer2 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block3_layer1 = block(in_planes=num_filters[1], planes=num_filters[2], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer2 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block4_layer1 = block(in_planes=num_filters[2], planes=num_filters[3], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block4_layer2 = block(in_planes=num_filters[3], planes=num_filters[3], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.deblock1 = self._make_delayer(num_filters[1], 1, 1)
        self.deblock2 = self._make_delayer(num_filters[2], 2, 2)
        self.deblock3 = self._make_delayer(num_filters[3], 4, 4)    
    def _make_delayer(self, plane, kernel, stride):
        delayers = [nn.ConvTranspose2d(plane, 128, kernel, stride=stride, bias=False),
                    nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                    nn.ReLU()]
        return nn.Sequential(*delayers)
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        add_features = data_dict['add_features']
        b11, add_features = self.block1_layer1(spatial_features, add_features)
        b12, add_features = self.block1_layer2(b11, add_features)
        b21, add_features = self.block2_layer1(b12, add_features)
        b22, add_features = self.block2_layer2(b21, add_features)
        b31, add_features = self.block3_layer1(b22, add_features)
        b32, add_features = self.block3_layer2(b31, add_features) 
        b41, add_features = self.block4_layer1(b32, add_features)
        b42, add_features = self.block4_layer2(b41, add_features) 
        d1 = self.deblock1(b22)
        d2 = self.deblock2(b32)
        d3 = self.deblock3(b42)
        return torch.cat([d1, d2, d3], dim=1)
class BaseBEVBackboneIntensityDopplerResnet34(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
        block = BasicBlockGeo 
        self.net = BackbonewithResnet34(block, model_cfg)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        x = self.net(data_dict)
        data_dict['spatial_features_2d'] = x
        return data_dict
class BackbonewithResnet34(nn.Module):
    def __init__(self, block, model_cfg) -> None:
        super().__init__()
        add_planes = 3
        self.model_cfg = model_cfg
        num_filters = [64, 64, 128, 256]
        
        self.block1_layer1 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block1_layer2 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block1_layer3 = block(in_planes=num_filters[0], planes=num_filters[0], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block2_layer1 = block(in_planes=num_filters[0], planes=num_filters[1], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block2_layer2 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block2_layer3 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block2_layer4 = block(in_planes=num_filters[1], planes=num_filters[1], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block3_layer1 = block(in_planes=num_filters[1], planes=num_filters[2], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer2 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer3 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer4 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer5 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block3_layer6 = block(in_planes=num_filters[2], planes=num_filters[2], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.block4_layer1 = block(in_planes=num_filters[2], planes=num_filters[3], stride=2, model_cfg=model_cfg, add_planes=add_planes)
        self.block4_layer2 = block(in_planes=num_filters[3], planes=num_filters[3], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        self.block4_layer3 = block(in_planes=num_filters[3], planes=num_filters[3], stride=1, model_cfg=model_cfg, add_planes=add_planes)
        
        self.deblock1 = self._make_delayer(num_filters[1], 1, 1)
        self.deblock2 = self._make_delayer(num_filters[2], 2, 2)
        self.deblock3 = self._make_delayer(num_filters[3], 4, 4)    
    def _make_delayer(self, plane, kernel, stride):
        delayers = [nn.ConvTranspose2d(plane, 128, kernel, stride=stride, bias=False),
                    nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                    nn.ReLU()]
        return nn.Sequential(*delayers)
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        add_features = data_dict['add_features']
        b11, add_features = self.block1_layer1(spatial_features, add_features)
        b12, add_features = self.block1_layer2(b11, add_features)
        b13, add_features = self.block1_layer3(b12, add_features)
        
        
        b21, add_features = self.block2_layer1(b13, add_features)
        b22, add_features = self.block2_layer2(b21, add_features)
        b23, add_features = self.block2_layer3(b22, add_features)
        b24, add_features = self.block2_layer4(b23, add_features)
        
        b31, add_features = self.block3_layer1(b24, add_features)
        b32, add_features = self.block3_layer2(b31, add_features) 
        b33, add_features = self.block3_layer3(b32, add_features) 
        b34, add_features = self.block3_layer4(b33, add_features) 
        b35, add_features = self.block3_layer5(b34, add_features) 
        b36, add_features = self.block3_layer6(b35, add_features) 
        
        
        b41, add_features = self.block4_layer1(b36, add_features)
        b42, add_features = self.block4_layer2(b41, add_features) 
        b43, add_features = self.block4_layer3(b42, add_features) 
        d1 = self.deblock1(b24)
        d2 = self.deblock2(b36)
        d3 = self.deblock3(b43)
        return torch.cat([d1, d2, d3], dim=1)


class BasicBlockGeo(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, model_cfg=None, add_planes=0):
        super(BasicBlockGeo, self).__init__()
        self.model_cfg = model_cfg
                
        self.conv1 = nn.Conv2d(in_planes + add_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        
        self.feature_max_pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        
        self.conv2 = nn.Conv2d(planes + add_planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        
        self.attention = nn.Sequential()
        if self.model_cfg.get('ATTENTION', None) is not None:
            if str(self.model_cfg.ATTENTION) == 'SE':
                self.attention = SELayer(self.expansion*planes)
            elif str(self.model_cfg.ATTENTION) == 'CBAM':
                self.attention = CBAM(self.expansion*planes)
            
        self.stride = stride
        
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        #     ) if str(self.model_cfg.RESNET) == 'A' else  nn.Sequential(
        #                                             nn.AvgPool2d(kernel_size=2, stride=stride),
        #                                             nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
        #                                             nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
        #                                         )
    def forward(self, x, add_features=None):
        identity = x
        add_features_max = None
        if add_features is not None:
            x = torch.cat([x, add_features], dim=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        if add_features is not None:
            add_features_max = self.feature_max_pool(add_features)
            out = torch.cat([out, add_features_max], dim=1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        # if self.shortcut is not None:
        #     out += self.shortcut(identity)
        out = self.relu2(out)
        return out, add_features_max

# class Bottleneck(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1, model_cfg=None):
#         super(Bottleneck, self).__init__()
#         self.model_cfg = model_cfg
 
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
#         self.relu1 = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
#         self.relu2 = nn.ReLU(inplace=True)
        
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
#         self.relu3 = nn.ReLU(inplace=True)
        
#         if self.model_cfg.get('ATTENTION', None) is not None:
#             if str(self.model_cfg.ATTENTION) == 'SE':
#                 self.attention = SELayer(self.expansion*planes)
#             elif str(self.model_cfg.ATTENTION) == 'CBAM':
#                 self.attention = CBAM(self.expansion*planes)
#         else:
#             self.attention = nn.Sequential()
        
#         self.stride = stride

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
#             ) if str(self.model_cfg.RESNET) == 'A' else  nn.Sequential(
#                                                     nn.AvgPool2d(in_planes, kernel_size=2, stride=stride),
#                                                     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
#                                                     nn.BatchNorm2d(self.expansion*planes, eps=1e-3, momentum=0.01)
#                                                 )
        
#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = self.attention(out)
#         if self.shortcut is not None:
#             out += self.shortcut(x)
#         out = self.relu3(out)
#         return out