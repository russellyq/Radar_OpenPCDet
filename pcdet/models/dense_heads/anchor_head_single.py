# import numpy as np
# import torch.nn as nn
# import torch
# from .anchor_head_template import AnchorHeadTemplate, AnchorHeadTemplate_multi
# import torch.nn.functional as F


# class AnchorHeadSingle(AnchorHeadTemplate):
#     def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
#                  predict_boxes_when_training=True, **kwargs):
#         super().__init__(
#             model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training
#         )

#         self.num_anchors_per_location = sum(self.num_anchors_per_location)

#         self.conv_cls = nn.Conv2d(
#             input_channels, self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
#         self.conv_box = nn.Conv2d(
#             input_channels, self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )

#         if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
#             self.conv_dir_cls = nn.Conv2d(
#                 input_channels,
#                 self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )
#         else:
#             self.conv_dir_cls = None
#         self.init_weights()

#     def init_weights(self):
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

#     def forward(self, data_dict):
#         spatial_features_2d = data_dict['spatial_features_2d']
        
#         cls_preds = self.conv_cls(spatial_features_2d)
#         box_preds = self.conv_box(spatial_features_2d)

#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

#         self.forward_ret_dict['cls_preds'] = cls_preds
#         self.forward_ret_dict['box_preds'] = box_preds

#         if self.conv_dir_cls is not None:
#             dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
#         else:
#             dir_cls_preds = None

#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             self.forward_ret_dict.update(targets_dict)

#         if not self.training or self.predict_boxes_when_training:
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#             )
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             data_dict['batch_box_preds'] = batch_box_preds
#             data_dict['cls_preds_normalized'] = False

#         return data_dict

# def _upsample(x, y):
#     _,_,H,W = y.size()
#     return F.interpolate(x, size=(H,W), mode='bilinear')

# class AnchorHeadSingle_bev(AnchorHeadTemplate):
#     def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
#                  predict_boxes_when_training=True, **kwargs):
#         super().__init__(
#             model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training
#         )
#         # 每个点有3个尺度的个先验框  每个先验框都有两个方向（0度，90度）
#         self.num_anchors_per_location = sum(self.num_anchors_per_location)

#         self.conv_cls = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
#         self.conv_box = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )
        
#         self.conv_cls_bev = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
        
#         self.conv_box_bev = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )
#         self.softmax = nn.Softmax(dim=1)
        
        
#         if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
#             self.conv_dir_cls = nn.Conv2d(
#                 input_channels,
#                 2*self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )
#             self.conv_dir_cls_bev = nn.Conv2d(
#                 input_channels,
#                 2*self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )

            
            
#         else:
#             self.conv_dir_cls = None
#             self.conv_dir_cls_bev = None
#         self.init_weights()

#     def init_weights(self):
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
#         nn.init.constant_(self.conv_cls_bev.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box_bev.weight, mean=0, std=0.001)

#     def forward(self, data_dict):
#         spatial_features_2d = data_dict['spatial_features_2d'] # [N, Cz, Hy=160, Wx=160]
#         spatial_features_2d_bev = data_dict['spatial_features_2d_bev'] # [N, Cx=64, Hz=160, Wy=160]
      
#         # 每个坐标点上面6个先验框的类别预测 --> 
#         cls_preds_output = self.conv_cls(spatial_features_2d)
#         cls_preds_bev_output = self.conv_cls_bev(spatial_features_2d_bev)
#         cls_preds, cls_preds_conf = torch.chunk(cls_preds_output, 2, dim=1)
#         cls_preds_bev, cls_preds_bev_conf = torch.chunk(cls_preds_bev_output, 2, dim=1)
#         cls_preds_conf, cls_preds_bev_conf = torch.chunk(self.softmax(torch.cat((cls_preds_conf, cls_preds_bev_conf), dim=1)), 2, dim=1)
#         cls_preds = cls_preds*cls_preds_conf + cls_preds_bev*cls_preds_bev_conf
        
#         # 每个坐标点上面6个先验框的参数预测 --> 其中每个先验框需要预测7个参数，分别是（x, y, z, w, l, h, θ）
#         box_preds_output = self.conv_box(spatial_features_2d) 
#         box_preds_bev_output = self.conv_box_bev(spatial_features_2d_bev)
#         box_preds, box_preds_conf = torch.chunk(box_preds_output, 2, dim=1)
#         box_preds_bev, box_preds_bev_conf = torch.chunk(box_preds_bev_output, 2, dim=1)
#         box_preds_conf, box_preds_bev_conf = torch.chunk(self.softmax(torch.cat((box_preds_conf, box_preds_bev_conf), dim=1)), 2, dim=1)
#         box_preds = box_preds*box_preds_conf + box_preds_bev*box_preds_bev_conf

#         # 维度调整，将类别放置在最后一维度
#         # cls_preds = torch.cat((cls_preds, cls_preds_multi), dim=1)
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         # box_preds = torch.cat((box_preds, box_preds_multi), dim=1)
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        

#         self.forward_ret_dict['cls_preds'] = cls_preds
#         self.forward_ret_dict['box_preds'] = box_preds

#         if self.conv_dir_cls is not None:
#             dir_cls_preds_output = self.conv_dir_cls(spatial_features_2d) 
#             dir_cls_preds_bev_output = self.conv_dir_cls_bev(spatial_features_2d_bev)
#             dir_cls_preds, dir_cls_preds_conf = torch.chunk(dir_cls_preds_output, 2, dim=1)
#             dir_cls_preds_bev, dir_cls_preds_bev_conf = torch.chunk(dir_cls_preds_bev_output, 2, dim=1)
#             dir_cls_preds_conf, dir_cls_preds_bev_conf = torch.chunk(self.softmax(torch.cat((dir_cls_preds_conf, dir_cls_preds_bev_conf), dim=1)), 2, dim=1)
#             dir_cls_preds = dir_cls_preds*dir_cls_preds_conf + dir_cls_preds_bev*dir_cls_preds_bev_conf
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
            
#         else:
#             dir_cls_preds = None
#             # dir_cls_preds_multi = None

#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             self.forward_ret_dict.update(targets_dict)

#         if not self.training or self.predict_boxes_when_training:
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#             )
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             data_dict['batch_box_preds'] = batch_box_preds
#             data_dict['cls_preds_normalized'] = False

#         return data_dict
    
# class AnchorHeadSingle_multi(AnchorHeadTemplate):
#     def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
#                  predict_boxes_when_training=True, **kwargs):
#         super().__init__(
#             model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training
#         )
#         # 每个点有3个尺度的个先验框  每个先验框都有两个方向（0度，90度）
#         self.num_anchors_per_location = sum(self.num_anchors_per_location)

#         self.conv_cls = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
#         self.conv_box = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )
        
#         self.conv_cls_multi = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
        
#         self.conv_box_multi = nn.Conv2d(
#             input_channels, 2*self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )
#         self.softmax = nn.Softmax(dim=1)
        
        
#         if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
#             self.conv_dir_cls = nn.Conv2d(
#                 input_channels,
#                 2*self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )
#             self.conv_dir_cls_multi = nn.Conv2d(
#                 input_channels,
#                 2*self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )

            
            
#         else:
#             self.conv_dir_cls = None
#         self.init_weights()

#     def init_weights(self):
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
#         nn.init.constant_(self.conv_cls_multi.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box_multi.weight, mean=0, std=0.001)

#     def forward(self, data_dict):
#         spatial_features_2d = data_dict['spatial_features_2d'] # [N, Cz, Hy=160, Wx=160]
#         spatial_features_2d_multi = data_dict['spatial_features_2d_multi'] # [N, Cx=64, Hz=160, Wy=160]
#         # spatial_features_2d_multi = spatial_features_2d_multi.permute(0, 2, 3, 1).contiguous()  # [N, Hz, Wy, Cx]
#         # spatial_features_2d_multi = _upsample(spatial_features_2d_multi, spatial_features_2d)
        
#         # 每个坐标点上面6个先验框的类别预测 --> 
#         cls_preds_output = self.conv_cls(spatial_features_2d)
#         cls_preds_multi_output = self.conv_cls_multi(spatial_features_2d_multi)
#         cls_preds, cls_preds_conf = torch.chunk(cls_preds_output, 2, dim=1)
#         cls_preds_multi, cls_preds_multi_conf = torch.chunk(cls_preds_multi_output, 2, dim=1)
#         cls_preds_conf, cls_preds_multi_conf = torch.chunk(self.softmax(torch.cat((cls_preds_conf, cls_preds_multi_conf), dim=1)), 2, dim=1)
#         cls_preds = cls_preds*cls_preds_conf + cls_preds_multi*cls_preds_multi_conf
        
#         # 每个坐标点上面6个先验框的参数预测 --> 其中每个先验框需要预测7个参数，分别是（x, y, z, w, l, h, θ）
#         box_preds_output = self.conv_box(spatial_features_2d) 
#         box_preds_multi_output = self.conv_box_multi(spatial_features_2d_multi)
#         box_preds, box_preds_conf = torch.chunk(box_preds_output, 2, dim=1)
#         box_preds_multi, box_preds_multi_conf = torch.chunk(box_preds_multi_output, 2, dim=1)
#         box_preds_conf, box_preds_multi_conf = torch.chunk(self.softmax(torch.cat((box_preds_conf, box_preds_multi_conf), dim=1)), 2, dim=1)
#         box_preds = box_preds*box_preds_conf + box_preds_multi*box_preds_multi_conf

#         # 维度调整，将类别放置在最后一维度
#         # cls_preds = torch.cat((cls_preds, cls_preds_multi), dim=1)
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         # box_preds = torch.cat((box_preds, box_preds_multi), dim=1)
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        

#         self.forward_ret_dict['cls_preds'] = cls_preds
#         self.forward_ret_dict['box_preds'] = box_preds

#         if self.conv_dir_cls is not None:
#             dir_cls_preds_output = self.conv_dir_cls(spatial_features_2d) 
#             dir_cls_preds_multi_output = self.conv_dir_cls_multi(spatial_features_2d_multi)
#             dir_cls_preds, dir_cls_preds_conf = torch.chunk(dir_cls_preds_output, 2, dim=1)
#             dir_cls_preds_multi, dir_cls_preds_multi_conf = torch.chunk(dir_cls_preds_multi_output, 2, dim=1)
#             dir_cls_preds_conf, dir_cls_preds_multi_conf = torch.chunk(self.softmax(torch.cat((dir_cls_preds_conf, dir_cls_preds_multi_conf), dim=1)), 2, dim=1)
#             dir_cls_preds = dir_cls_preds*dir_cls_preds_conf + dir_cls_preds_multi*dir_cls_preds_multi_conf
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
            
#         else:
#             dir_cls_preds = None
#             # dir_cls_preds_multi = None

#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             self.forward_ret_dict.update(targets_dict)

#         if not self.training or self.predict_boxes_when_training:
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#             )
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             data_dict['batch_box_preds'] = batch_box_preds
#             data_dict['cls_preds_normalized'] = False

#         return data_dict
    

# class AnchorHeadSingle_multi_feature_cat(AnchorHeadTemplate):
#     def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
#                  predict_boxes_when_training=True, **kwargs):
#         super().__init__(
#             model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training
#         )
#         # 每个点有3个尺度的个先验框  每个先验框都有两个方向（0度，90度）
#         self.num_anchors_per_location = sum(self.num_anchors_per_location)

#         self.conv_cls = nn.Conv2d(
#             160+input_channels, self.num_anchors_per_location * self.num_class,
#             kernel_size=1
#         )
#         self.conv_box = nn.Conv2d(
#             160+input_channels, self.num_anchors_per_location * self.box_coder.code_size,
#             kernel_size=1
#         )
        
#         # self.conv_cls_multi = nn.Conv2d(
#         #     input_channels, self.num_anchors_per_location * self.num_class,
#         #     kernel_size=1
#         # )
        
#         # self.conv_box_multi = nn.Conv2d(
#         #     input_channels, self.num_anchors_per_location * self.box_coder.code_size,
#         #     kernel_size=1
#         # )
#         # self.softmax = nn.Softmax(dim=1)
        
        
#         if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
#             self.conv_dir_cls = nn.Conv2d(
#                 160+input_channels,
#                 self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                 kernel_size=1
#             )
#             # self.conv_dir_cls_multi = nn.Conv2d(
#             #     input_channels,
#             #     self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#             #     kernel_size=1
#             # )

            
            
#         else:
#             self.conv_dir_cls = None
#         self.init_weights()

#     def init_weights(self):
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
#         # nn.init.constant_(self.conv_cls_multi.bias, -np.log((1 - pi) / pi))
#         # nn.init.normal_(self.conv_box_multi.weight, mean=0, std=0.001)

#     def forward(self, data_dict):
#         spatial_features_2d = data_dict['spatial_features_2d'] # [N, 6C, H=160, W=160]
#         spatial_features_2d_multi = data_dict['spatial_features_2d_multi'] # [N, 6C, H=160, W=160]
#         # print('\n spatial_features_2d_multi shape:, ', spatial_features_2d_multi.shape)
#         spatial_features_2d_multi = spatial_features_2d_multi.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         spatial_features_2d_multi = _upsample(spatial_features_2d_multi, spatial_features_2d)
#         spatial_features_2d = torch.cat((spatial_features_2d, spatial_features_2d_multi), dim=1)
        
#         # 每个坐标点上面6个先验框的类别预测 --> 
#         cls_preds = self.conv_cls(spatial_features_2d)
   
#         # 每个坐标点上面6个先验框的参数预测 --> 其中每个先验框需要预测7个参数，分别是（x, y, z, w, l, h, θ）
#         box_preds = self.conv_box(spatial_features_2d) 
#         # box_preds_multi = self.conv_box_multi(spatial_features_2d_multi)

#         # 维度调整，将类别放置在最后一维度
#         # cls_preds = torch.cat((cls_preds, cls_preds_multi), dim=1)
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         # cls_preds_multi = cls_preds_multi.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         # box_preds = torch.cat((box_preds, box_preds_multi), dim=1)
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#         # box_preds_multi = box_preds_multi.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        

#         self.forward_ret_dict['cls_preds'] = cls_preds
#         # self.forward_ret_dict['cls_preds_multi'] = cls_preds_multi
#         self.forward_ret_dict['box_preds'] = box_preds
#         # self.forward_ret_dict['box_preds_multi'] = box_preds_multi

#         if self.conv_dir_cls is not None:
#             dir_cls_preds = self.conv_dir_cls(spatial_features_2d) 
#             # dir_cls_preds_multi = self.conv_dir_cls_multi(spatial_features_2d_multi)
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             # dir_cls_preds_multi = dir_cls_preds_multi.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
#             # self.forward_ret_dict['dir_cls_preds_multi'] = dir_cls_preds_multi
            
#         else:
#             dir_cls_preds = None
#             # dir_cls_preds_multi = None

#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             self.forward_ret_dict.update(targets_dict)

#         if not self.training or self.predict_boxes_when_training:
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#             )
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             data_dict['batch_box_preds'] = batch_box_preds
#             data_dict['cls_preds_normalized'] = False

#         return data_dict
import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        print('point_cloud_range:,', point_cloud_range)
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
