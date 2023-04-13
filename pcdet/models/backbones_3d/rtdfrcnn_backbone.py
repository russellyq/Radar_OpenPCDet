from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from pcdet.models.backbones_2d.base_2D_backbone import ImageBackbone
import math

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    """
    后处理执行块，根据conv_type选择对应的卷积操作并和norm与激活函数封装为块
    """
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class RTDFRCNNBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        point_cloud_range=[-2, -25.6, 0, 2, 25.6, 51.2]
        voxel_size = [0.1, 0.05, 0.05]
        
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # [41, 1600, 1408] 在原始网格的高度方向上增加了一维

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16 + 16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32 + 32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64 + 64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64 + 64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16 + 16,
            'x_conv2': 32 + 32,
            'x_conv3': 64 + 64,
            'x_conv4': 64 + 64
        }
        
        
        
        self.image_conv = ImageBackbone()
        self.attention1 = DotProductAttention(16, 16, 16)
        self.attention2 = DotProductAttention(32, 32, 32)
        self.attention3 = DotProductAttention(64, 64, 64)
        self.attention4 = DotProductAttention(64, 64, 64)

    
    def process(self, sp_tensor, img_tensor, attention_head, batch_dict, cur_stride):
        h, w = batch_dict['images'].shape[2:]
        batch_index = sp_tensor.indices[:, 0]
        spatial_indices = sp_tensor.indices[:, 1:] * cur_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        img_features, lidar_features, img_lidar_features, lidar_indices = [], [], [], []
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        for batch_id in range(batch_size):
            img_rgb_batch = img_tensor[batch_id]
            calib = calibs[batch_id]
            voxels_3d_batch = voxels_3d[batch_index==batch_id]
            voxel_features_sparse = sp_tensor.features[batch_index==batch_id]
            voxel_indices_sparse = sp_tensor.indices[batch_index==batch_id]
            
            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][batch_id]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_id].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][batch_id] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][batch_id] else 1
        
            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(img_rgb_batch.device).long()
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], img_rgb_batch.shape[0]), device=img_rgb_batch.device)
            image_features_batch[filter_idx] = img_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            # image_with_voxelfeatures_batch  = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
            image_with_voxelfeatures_batch = attention_head(voxel_features_sparse, image_features_batch, voxels_3d_batch[:, self.inv_idx])
            
            img_lidar_features.append(image_with_voxelfeatures_batch)
            lidar_indices.append(voxel_indices_sparse)
        
        lidar_indices = torch.cat(lidar_indices)
        img_lidar_features = torch.cat(img_lidar_features)
        
    
        return spconv.SparseConvTensor(img_lidar_features, lidar_indices, sp_tensor.spatial_shape, sp_tensor.batch_size)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        imgs = batch_dict['images']  
        img_conv1, img_conv2, img_conv3, img_conv4 = self.image_conv(imgs) # 1, 1/2, 1/4, 1/8
        
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x) 
        x_conv1 = self.process(x_conv1, img_conv1, self.attention1, batch_dict, cur_stride=1)
        # x_conv1: 16 + 16
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.process(x_conv2, img_conv2, self.attention2, batch_dict, cur_stride=2)
        
        # x_conv2: 32 + 16
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.process(x_conv3, img_conv3, self.attention3, batch_dict, cur_stride=4)

        
        # x_conv3: 64 + 16
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.process(x_conv4, img_conv4, self.attention4, batch_dict, cur_stride=8)

        
        # x_conv4: 64 + 16
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
         # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # 多尺度特征
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # 多尺度下采样倍数
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class DotProductAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels, qkv_channels):
        super().__init__()
        # self.query = nn.Linear(in_features=lidar_channels, out_features=qkv_channels, bias=False)
        # self.key = nn.Linear(in_features=3, out_features=3, bias=False)
        # self.value = nn.Linear(in_features=lidar_channels, out_features=qkv_channels, bias=False)
        
        self.headattention = nn.MultiheadAttention(qkv_channels, 1, batch_first=True, kdim=3, vdim=qkv_channels)
        self.fc = nn.Linear(in_features=qkv_channels, out_features=lidar_channels, bias=False)
        
    def forward(self, lidar, camera, lidar_coord):
        # lidar, camera: N * C
        query = camera.unsqueeze(0)
        key = lidar_coord.unsqueeze(0)
        value = lidar.unsqueeze(0)
        
        attn_output, _ = self.headattention(query, key, value)
        attn_output = F.relu(self.fc(attn_output.squeeze(0)))
        out = torch.cat((lidar, attn_output), dim=1)
        return out

class RTDFRCNNBackBone_Concat(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        point_cloud_range=[-2, -25.6, 0, 2, 25.6, 51.2]
        voxel_size = [0.1, 0.05, 0.05]
        
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # [41, 1600, 1408] 在原始网格的高度方向上增加了一维

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16 + 16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32 + 32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64 + 64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64 + 64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16 + 16,
            'x_conv2': 32 + 32,
            'x_conv3': 64 + 64,
            'x_conv4': 64 + 64
        }
        
        
        
        self.image_conv = ImageBackbone()


    
    def process(self, sp_tensor, img_tensor, batch_dict, cur_stride):
        h, w = batch_dict['images'].shape[2:]
        batch_index = sp_tensor.indices[:, 0]
        spatial_indices = sp_tensor.indices[:, 1:] * cur_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        img_features, lidar_features, img_lidar_features, lidar_indices = [], [], [], []
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        for batch_id in range(batch_size):
            img_rgb_batch = img_tensor[batch_id]
            calib = calibs[batch_id]
            voxels_3d_batch = voxels_3d[batch_index==batch_id]
            voxel_features_sparse = sp_tensor.features[batch_index==batch_id]
            voxel_indices_sparse = sp_tensor.indices[batch_index==batch_id]
            
            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][batch_id]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_id].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][batch_id] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][batch_id] else 1
        
            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(img_rgb_batch.device).long()
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], img_rgb_batch.shape[0]), device=img_rgb_batch.device)
            image_features_batch[filter_idx] = img_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            image_with_voxelfeatures_batch  = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
            
            img_lidar_features.append(image_with_voxelfeatures_batch)
            lidar_indices.append(voxel_indices_sparse)
        
        lidar_indices = torch.cat(lidar_indices)
        img_lidar_features = torch.cat(img_lidar_features)
        
    
        return spconv.SparseConvTensor(img_lidar_features, lidar_indices, sp_tensor.spatial_shape, sp_tensor.batch_size)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        imgs = batch_dict['images']  
        img_conv1, img_conv2, img_conv3, img_conv4 = self.image_conv(imgs) # 1, 1/2, 1/4, 1/8
        
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x) 
        x_conv1 = self.process(x_conv1, img_conv1, batch_dict, cur_stride=1)
        # x_conv1: 16 + 16
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.process(x_conv2, img_conv2, batch_dict, cur_stride=2)
        
        # x_conv2: 32 + 16
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.process(x_conv3, img_conv3, batch_dict, cur_stride=4)

        
        # x_conv3: 64 + 16
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.process(x_conv4, img_conv4, batch_dict, cur_stride=8)

        
        # x_conv4: 64 + 16
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
         # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # 多尺度特征
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # 多尺度下采样倍数
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class RTDFRCNNBackBone_Sum(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        point_cloud_range=[-2, -25.6, 0, 2, 25.6, 51.2]
        voxel_size = [0.1, 0.05, 0.05]
        
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # [41, 1600, 1408] 在原始网格的高度方向上增加了一维

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        
        
        
        self.image_conv = ImageBackbone()


    
    def process(self, sp_tensor, img_tensor, batch_dict, cur_stride):
        h, w = batch_dict['images'].shape[2:]
        batch_index = sp_tensor.indices[:, 0]
        spatial_indices = sp_tensor.indices[:, 1:] * cur_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        img_features, lidar_features, img_lidar_features, lidar_indices = [], [], [], []
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        for batch_id in range(batch_size):
            img_rgb_batch = img_tensor[batch_id]
            calib = calibs[batch_id]
            voxels_3d_batch = voxels_3d[batch_index==batch_id]
            voxel_features_sparse = sp_tensor.features[batch_index==batch_id]
            voxel_indices_sparse = sp_tensor.indices[batch_index==batch_id]
            
            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][batch_id]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_id].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][batch_id] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][batch_id] else 1
        
            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(img_rgb_batch.device).long()
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], img_rgb_batch.shape[0]), device=img_rgb_batch.device)
            image_features_batch[filter_idx] = img_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            image_with_voxelfeatures_batch  = torch.add(image_features_batch, voxel_features_sparse)
            
            img_lidar_features.append(image_with_voxelfeatures_batch)
            lidar_indices.append(voxel_indices_sparse)
        
        lidar_indices = torch.cat(lidar_indices)
        img_lidar_features = torch.cat(img_lidar_features)
        
    
        return spconv.SparseConvTensor(img_lidar_features, lidar_indices, sp_tensor.spatial_shape, sp_tensor.batch_size)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        imgs = batch_dict['images']  
        img_conv1, img_conv2, img_conv3, img_conv4 = self.image_conv(imgs) # 1, 1/2, 1/4, 1/8
        
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x) 
        x_conv1 = self.process(x_conv1, img_conv1, batch_dict, cur_stride=1)
        # x_conv1: 16 + 16
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.process(x_conv2, img_conv2, batch_dict, cur_stride=2)
        
        # x_conv2: 32 + 16
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.process(x_conv3, img_conv3, batch_dict, cur_stride=4)

        
        # x_conv3: 64 + 16
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.process(x_conv4, img_conv4, batch_dict, cur_stride=8)

        
        # x_conv4: 64 + 16
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
         # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # 多尺度特征
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # 多尺度下采样倍数
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class RTDFRCNNBackBone_DeepFusionAttention(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        point_cloud_range=[-2, -25.6, 0, 2, 25.6, 51.2]
        voxel_size = [0.1, 0.05, 0.05]
        
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # [41, 1600, 1408] 在原始网格的高度方向上增加了一维

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16 + 16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32 + 32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64 + 64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64 + 64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16 + 16,
            'x_conv2': 32 + 32,
            'x_conv3': 64 + 64,
            'x_conv4': 64 + 64
        }
        
        
        
        self.image_conv = ImageBackbone()
        self.attention1 = DeepFusionAttention(16, 16, 16)
        self.attention2 = DeepFusionAttention(32, 32, 32)
        self.attention3 = DeepFusionAttention(64, 64, 64)
        self.attention4 = DeepFusionAttention(64, 64, 64)

    
    def process(self, sp_tensor, img_tensor, attention_head, batch_dict, cur_stride):
        h, w = batch_dict['images'].shape[2:]
        batch_index = sp_tensor.indices[:, 0]
        spatial_indices = sp_tensor.indices[:, 1:] * cur_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        img_features, lidar_features, img_lidar_features, lidar_indices = [], [], [], []
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        for batch_id in range(batch_size):
            img_rgb_batch = img_tensor[batch_id]
            calib = calibs[batch_id]
            voxels_3d_batch = voxels_3d[batch_index==batch_id]
            voxel_features_sparse = sp_tensor.features[batch_index==batch_id]
            voxel_indices_sparse = sp_tensor.indices[batch_index==batch_id]
            
            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][batch_id]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_id].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][batch_id] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][batch_id] else 1
        
            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(img_rgb_batch.device).long()
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], img_rgb_batch.shape[0]), device=img_rgb_batch.device)
            image_features_batch[filter_idx] = img_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            # image_with_voxelfeatures_batch  = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
            image_with_voxelfeatures_batch = attention_head(voxel_features_sparse, image_features_batch)
            
            img_lidar_features.append(image_with_voxelfeatures_batch)
            lidar_indices.append(voxel_indices_sparse)
        
        lidar_indices = torch.cat(lidar_indices)
        img_lidar_features = torch.cat(img_lidar_features)
        
    
        return spconv.SparseConvTensor(img_lidar_features, lidar_indices, sp_tensor.spatial_shape, sp_tensor.batch_size)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        imgs = batch_dict['images']  
        img_conv1, img_conv2, img_conv3, img_conv4 = self.image_conv(imgs) # 1, 1/2, 1/4, 1/8
        
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x) 
        x_conv1 = self.process(x_conv1, img_conv1, self.attention1, batch_dict, cur_stride=1)
        # x_conv1: 16 + 16
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.process(x_conv2, img_conv2, self.attention2, batch_dict, cur_stride=2)
        
        # x_conv2: 32 + 16
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.process(x_conv3, img_conv3, self.attention3, batch_dict, cur_stride=4)

        
        # x_conv3: 64 + 16
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.process(x_conv4, img_conv4, self.attention4, batch_dict, cur_stride=8)

        
        # x_conv4: 64 + 16
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
         # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # 多尺度特征
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # 多尺度下采样倍数
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class DeepFusionAttention(nn.Module):
    def __init__(self, lidar_channels, image_channels, qkv_channels):
        super().__init__()
        self.query = nn.Linear(in_features=lidar_channels, out_features=qkv_channels, bias=False)
        self.key = nn.Linear(in_features=image_channels, out_features=qkv_channels, bias=False)
        self.value = nn.Linear(in_features=image_channels, out_features=qkv_channels, bias=False)
        
        self.headattention = nn.MultiheadAttention(qkv_channels, 1, batch_first=True)
        self.fc = nn.Linear(in_features=qkv_channels, out_features=lidar_channels, bias=False)
        
    def forward(self, lidar, camera):
        # lidar, camera: N * C
        query = self.query(lidar.unsqueeze(0))
        key = self.key(camera.unsqueeze(0))
        value = self.value(camera.unsqueeze(0))
        
        attn_output, _ = self.headattention(query, key, value)
        attn_output = F.relu(self.fc(attn_output.squeeze(0)))
        out = torch.cat((lidar, attn_output), dim=1)
        return out

class RTDFRCNNBackBone_ab_study(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        point_cloud_range=[-2, -25.6, 0, 2, 25.6, 51.2]
        voxel_size = [0.1, 0.05, 0.05]
        
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # [41, 1600, 1408] 在原始网格的高度方向上增加了一维

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32+32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64 + 64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64 + 64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32+32,
            'x_conv3': 64 + 64,
            'x_conv4': 64 + 64
        }
        
        
        
        self.image_conv = ImageBackbone()
        # self.attention1 = DotProductAttention(16, 16, 16)
        self.attention2 = DotProductAttention(32, 32, 32)
        self.attention3 = DotProductAttention(64, 64, 64)
        self.attention4 = DotProductAttention(64, 64, 64)

    
    def process(self, sp_tensor, img_tensor, attention_head, batch_dict, cur_stride):
        h, w = batch_dict['images'].shape[2:]
        batch_index = sp_tensor.indices[:, 0]
        spatial_indices = sp_tensor.indices[:, 1:] * cur_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        img_features, lidar_features, img_lidar_features, lidar_indices = [], [], [], []
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        for batch_id in range(batch_size):
            img_rgb_batch = img_tensor[batch_id]
            calib = calibs[batch_id]
            voxels_3d_batch = voxels_3d[batch_index==batch_id]
            voxel_features_sparse = sp_tensor.features[batch_index==batch_id]
            voxel_indices_sparse = sp_tensor.indices[batch_index==batch_id]
            
            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][batch_id]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_id].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][batch_id] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][batch_id] else 1
        
            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(img_rgb_batch.device).long()
            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], img_rgb_batch.shape[0]), device=img_rgb_batch.device)
            image_features_batch[filter_idx] = img_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            # image_with_voxelfeatures_batch  = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
            image_with_voxelfeatures_batch = attention_head(voxel_features_sparse, image_features_batch, voxels_3d_batch[:, self.inv_idx])
            
            img_lidar_features.append(image_with_voxelfeatures_batch)
            lidar_indices.append(voxel_indices_sparse)
        
        lidar_indices = torch.cat(lidar_indices)
        img_lidar_features = torch.cat(img_lidar_features)
        
    
        return spconv.SparseConvTensor(img_lidar_features, lidar_indices, sp_tensor.spatial_shape, sp_tensor.batch_size)


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        imgs = batch_dict['images']  
        img_conv1, img_conv2, img_conv3, img_conv4 = self.image_conv(imgs) # 1, 1/2, 1/4, 1/8
        
        # 根据voxel特征和坐标以及空间形状和batch，建立稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # 始终以SparseConvTensor的形式输出
        # 主要包括:
        # batch_size: batch size大小
        # features: (特征数量，特征维度)
        # indices: (特征数量，特征索引(4维，第一维度是batch索引))
        # spatial_shape:(z,y,x)
        # indice_dict{(tuple:5),}:0:输出索引，1:输入索引，2:输入Rulebook索引，3:输出Rulebook索引，4:spatial shape
        # sparity:稀疏率
        # 在heigh_compression.py中结合batch，spatial_shape、indice和feature将特征还原的对应位置，并在高度方向合并压缩至BEV特征图

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x) 
        # x_conv1 = self.process(x_conv1, img_conv1, self.attention1, batch_dict, cur_stride=1)
        # x_conv1: 16 + 16
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.process(x_conv2, img_conv2, self.attention2, batch_dict, cur_stride=2)
        
        # x_conv2: 32 + 16
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.process(x_conv3, img_conv3, self.attention3, batch_dict, cur_stride=4)

        
        # x_conv3: 64 + 16
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.process(x_conv4, img_conv4, self.attention4, batch_dict, cur_stride=8)

        
        # x_conv4: 64 + 16
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
         # 将输出特征图和各尺度的3d特征图存入batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # 多尺度特征
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # 多尺度下采样倍数
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict