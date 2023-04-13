import numpy as np
import torch
import torch.nn as nn
from .attention_layer import SELayer, DeformableConv2d, CBAM
class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

# class BaseBEVBackbone(nn.Module):
#     def __init__(self, model_cfg, input_channels):
#         super().__init__()
#         self.model_cfg = model_cfg
        
#         conv = nn.Conv2d if self.model_cfg.DEFORMABLE==False else DeformableConv2d
#         se_attention = SELayer
#         cbam = CBAM

#         if self.model_cfg.get('LAYER_NUMS', None) is not None:
#             assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
#             layer_nums = self.model_cfg.LAYER_NUMS
#             layer_strides = self.model_cfg.LAYER_STRIDES
#             num_filters = self.model_cfg.NUM_FILTERS
#         else:
#             layer_nums = layer_strides = num_filters = []

#         if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
#             assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
#             num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
#             upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
#         else:
#             upsample_strides = num_upsample_filters = []

#         num_levels = len(layer_nums)
#         c_in_list = [input_channels, *num_filters[:-1]]
#         self.blocks = nn.ModuleList()
#         self.deblocks = nn.ModuleList()
#         for idx in range(num_levels):
#             cur_layers = [
#                 nn.ZeroPad2d(1),
#                 # nn.Conv2d(
#                 #     c_in_list[idx], num_filters[idx], kernel_size=3,
#                 #     stride=layer_strides[idx], padding=0, bias=False
#                 # ),
#                 conv(
#                     c_in_list[idx], num_filters[idx], kernel_size=3,
#                     stride=layer_strides[idx], padding=0, bias=False
#                 ),
#                 nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                 nn.ReLU()
#             ]
#             for k in range(layer_nums[idx]):
#                 if self.model_cfg.SE_ATTENTION == True:
#                     cur_layers.extend([
#                         se_attention(num_filters[idx]),
#                         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ])

#                 elif self.model_cfg.CBAM == True:
#                     cur_layers.extend([
#                         cbam(num_filters[idx]),
#                         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
                        
#                     ])
#                 else:
#                     cur_layers.extend([
#                         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ])
                 
#             self.blocks.append(nn.Sequential(*cur_layers))
#             if len(upsample_strides) > 0:
#                 stride = upsample_strides[idx]
#                 if stride >= 1:
#                     self.deblocks.append(nn.Sequential(
#                         nn.ConvTranspose2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             upsample_strides[idx],
#                             stride=upsample_strides[idx], bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))
#                 else:
#                     stride = np.round(1 / stride).astype(np.int)
#                     self.deblocks.append(nn.Sequential(
#                         nn.Conv2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             stride,
#                             stride=stride, bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))

#         c_in = sum(num_upsample_filters)
#         if len(upsample_strides) > num_levels:
#             self.deblocks.append(nn.Sequential(
#                 nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
#                 nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
#                 nn.ReLU(),
#             ))

#         self.num_bev_features = c_in

#     def forward(self, data_dict):
#         """
#         Args:
#             data_dict:
#                 spatial_features
#         Returns:
#         """
#         spatial_features = data_dict['spatial_features']
#         ups = []
#         ret_dict = {}
#         x = spatial_features
#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x)

#             stride = int(spatial_features.shape[2] / x.shape[2])
#             ret_dict['spatial_features_%dx' % stride] = x
#             if len(self.deblocks) > 0:
#                 ups.append(self.deblocks[i](x))
#             else:
#                 ups.append(x)

#         if len(ups) > 1:
#             x = torch.cat(ups, dim=1)
#         elif len(ups) == 1:
#             x = ups[0]

#         if len(self.deblocks) > len(self.blocks):
#             x = self.deblocks[-1](x)

#         data_dict['spatial_features_2d'] = x

#         return data_dict
    

class BaseBEVBackbone_bev(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        conv = nn.Conv2d if self.model_cfg.DEFORMABLE==False else DeformableConv2d
        se_attention = SELayer
        cbam = CBAM

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.bev_prepare = BasicBlock(6, 64, 1)
        self.blocks, self.blocks_bev = nn.ModuleList(), nn.ModuleList()
        self.deblocks, self.deblocks_bev = nn.ModuleList(), nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                # nn.Conv2d(
                #     c_in_list[idx], num_filters[idx], kernel_size=3,
                #     stride=layer_strides[idx], padding=0, bias=False
                # ),
                conv(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                if self.model_cfg.SE_ATTENTION == True:
                    cur_layers.extend([
                        se_attention(num_filters[idx]),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])

                elif self.model_cfg.CBAM == True:
                    cur_layers.extend([
                        cbam(num_filters[idx]),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                        
                    ])
                else:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                 
            self.blocks.append(nn.Sequential(*cur_layers))
            self.blocks_bev.append(nn.Sequential(*cur_layers))
            
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_bev.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            (upsample_strides[idx], upsample_strides[idx]),
                            stride=(upsample_strides[idx], upsample_strides[idx]), bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_bev.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            (stride, stride),
                            stride=(stride,stride), bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_bev.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, (upsample_strides[-1], upsample_strides[-1]), stride=(upsample_strides[-1], upsample_strides[-1]), bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features, spatial_features_multi = data_dict['spatial_features'], data_dict['bev']
        # print('\n spatial_features: ', spatial_features.shape)
        # print('\n spatial_features_multi: ', spatial_features_multi.shape)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        
        ups = []
        ret_dict = {}
        x = spatial_features_multi
        x = self.bev_prepare(x)
        for i in range(len(self.blocks_bev)):
            x = self.blocks_bev[i](x)

            stride = int(spatial_features_multi.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks_bev) > 0:
                ups.append(self.deblocks_bev[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks_bev) > len(self.blocks_bev):
            x = self.deblocks_bev[-1](x)

        data_dict['spatial_features_2d_bev'] = x
        
        return data_dict 

class BaseBEVBackbone_multiview(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        conv = nn.Conv2d if self.model_cfg.DEFORMABLE==False else DeformableConv2d
        se_attention = SELayer
        cbam = CBAM

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.blocks,  self.blocks_multiview = nn.ModuleList(), nn.ModuleList()
        self.deblocks, self.deblocks_multiview = nn.ModuleList(), nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                # nn.Conv2d(
                #     c_in_list[idx], num_filters[idx], kernel_size=3,
                #     stride=layer_strides[idx], padding=0, bias=False
                # ),
                conv(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                if self.model_cfg.SE_ATTENTION == True:
                    cur_layers.extend([
                        se_attention(num_filters[idx]),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])

                elif self.model_cfg.CBAM == True:
                    cur_layers.extend([
                        cbam(num_filters[idx]),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                        
                    ])
                else:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                 
            self.blocks.append(nn.Sequential(*cur_layers))
            self.blocks_multiview.append(nn.Sequential(*cur_layers))
            
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_multiview.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            (8*upsample_strides[idx], upsample_strides[idx]),
                            stride=(8*upsample_strides[idx], upsample_strides[idx]), bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_multiview.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            (8*stride, stride),
                            stride=(8*stride,stride), bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_multiview.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, (8*upsample_strides[-1], upsample_strides[-1]), stride=(8*upsample_strides[-1], upsample_strides[-1]), bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features, spatial_features_multi = data_dict['spatial_features'], data_dict['spatial_features_multi']
        # print('\n spatial_features: ', spatial_features.shape)
        # print('\n spatial_features_multi: ', spatial_features_multi.shape)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        
        ups = []
        ret_dict = {}
        x = spatial_features_multi
        for i in range(len(self.blocks_multiview)):
            x = self.blocks_multiview[i](x)

            stride = int(spatial_features_multi.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks_multiview) > 0:
                ups.append(self.deblocks_multiview[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks_multiview) > len(self.blocks_multiview):
            x = self.deblocks_multiview[-1](x)

        data_dict['spatial_features_2d_multi'] = x
        
        return data_dict 
    
class BaseBEVBackbone_multiview_projection(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.blocks,  self.blocks_multiview = nn.ModuleList(), nn.ModuleList()
        self.deblocks, self.deblocks_multiview = nn.ModuleList(), nn.ModuleList()
        
        self.prepare = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                 
            self.blocks.append(nn.Sequential(*cur_layers))
            self.blocks_multiview.append(nn.Sequential(*cur_layers))
            
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_multiview.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_multiview.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride, 
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_multiview.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features, proj_feature = data_dict['spatial_features'], data_dict['proj_feature']
        # print('\n spatial_features: ', spatial_features.shape)
        # print('\n spatial_features_multi: ', spatial_features_multi.shape)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        
        ups = []
        ret_dict = {}
        proj_feature = self.prepare(proj_feature)
        x = proj_feature
        for i in range(len(self.blocks_multiview)):
            x = self.blocks_multiview[i](x)

            stride = int(proj_feature.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks_multiview) > 0:
                ups.append(self.deblocks_multiview[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks_multiview) > len(self.blocks_multiview):
            x = self.deblocks_multiview[-1](x)

        data_dict['spatial_features_2d_multi'] = x
        
        return data_dict

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
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