from select import select
import torch
import torch.nn as nn
from pcdet.models.backbones_3d.vfe.pillar_vfe import PointNet
import numpy as np
class PointPillarScatter(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES  # 64
        self.nx, self.ny, self.nz = grid_size  # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            pillar_features:(M=31530,64)
            coords:(M, 4) 第一维是batch_index 其余维度为xyz
        Returns:
            batch_spatial_features:(batch_size, 64, 496, 432)
        """
        # 拿到经过前面pointnet处理过后的pillar数据和每个pillar所在点云中的坐标位置
        # pillar_features 维度 （M， 64）
        # coords 维度 （M， 4）
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        # 将转换成为伪图像的数据存在到该列表中
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        # batch中的每个数据独立处理
        for batch_idx in range(batch_size):
            # 创建一个空间坐标所有用来接受pillar中的数据
            # self.num_bev_features是64
            # self.nz * self.nx * self.ny是生成的空间坐标索引 [496, 432, 1]的乘积
            # spatial_feature 维度 (64,214272)
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)  # (64,214272)-->1x432x496=214272

            # 从coords[:, 0]取出该batch_idx的数据mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :]
            # this_coords中存储的坐标是z,y和x的形式,且只有一层，因此计算索引的方式如下
            # 平铺后需要计算前面有多少个pillar 一直到当前pillar的索引
            """
            因为前面是将所有数据flatten成一维的了，相当于一个图片宽高为[496, 432]的图片
            被flatten成一维的图片数据了，变成了496*432=214272;
            而this_coords中存储的是平面（不需要考虑Z轴）中一个点的信息，所以要
            将这个点的位置放回被flatten的一位数据时，需要计算在该点之前所有行的点总和加上
            该点所在的列即可
            """
            # 这里得到所有非空pillar在伪图像的对应索引位置
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, 214272)
            batch_spatial_features.append(spatial_feature)

        # 在第0个维度将所有的数据堆叠在一起
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # reshape回原空间(伪图像)    （4, 64, 214272）--> (4, 64, 496, 432)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        # 将结果加入batch_dict
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter_addfeatures(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_add_features = 3
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        

    def forward(self, batch_dict, **kwargs):
        # pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        add_features_to_map, pillar_features, coords = batch_dict['add_features_to_map'], batch_dict['pillar_features'], batch_dict['voxel_coords']
        # spatial features
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        # add_features
        batch_add_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            add_feature = torch.zeros(
                self.num_bev_add_features,
                self.nz * self.nx * self.ny,
                dtype=add_features_to_map.dtype,
                device=add_features_to_map.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = add_features_to_map[batch_mask, :]
            pillars = pillars.t()
            add_feature[:, indices] = pillars
            batch_add_features.append(add_feature)

        batch_add_features = torch.stack(batch_add_features, 0)
        batch_add_features = batch_add_features.view(batch_size, self.num_bev_add_features * self.nz, self.ny, self.nx)
        batch_dict['add_features'] = batch_add_features
        
        return batch_dict
    
class PointPillarScatter_Multi(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_add_features = 3
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        self.nx_multi, self.ny_multi, self.nz_multi = 1, 320, 40
        

    def forward(self, batch_dict, **kwargs):
        # pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        add_features_to_map, pillar_features, coords = batch_dict['add_features_to_map'], batch_dict['pillar_features'], batch_dict['voxel_coords']
        if 'pillar_features_multi' in batch_dict and 'voxel_coords_multi' in batch_dict:
            pillar_features_multi, coords_multi = batch_dict['pillar_features_multi'], batch_dict['voxel_coords_multi']
            # spatial_feature_multi
            batch_spatial_features_multi = []
            batch_size = coords_multi[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz_multi * self.nx_multi * self.ny_multi,
                    dtype=pillar_features_multi.dtype,
                    device=pillar_features_multi.device)

                batch_mask = coords_multi[:, 0] == batch_idx
                this_coords = coords_multi[batch_mask, :]
                indices = this_coords[:, 1] * self.ny_multi + this_coords[:, 2]  + this_coords[:, 3]
                indices = indices.type(torch.long)
                pillars = pillar_features_multi[batch_mask, :]
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
                batch_spatial_features_multi.append(spatial_feature)

            batch_spatial_features_multi = torch.stack(batch_spatial_features_multi, 0)
            batch_spatial_features_multi = batch_spatial_features_multi.view(batch_size, self.num_bev_features*self.nx_multi, self.nz_multi, self.ny_multi)
            batch_dict['spatial_features_multi'] = batch_spatial_features_multi
            # print('\nspatial_features_multi, ', batch_spatial_features_multi.shape)
        
        # spatial features
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        # add_features
        batch_add_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            add_feature = torch.zeros(
                self.num_bev_add_features,
                self.nz * self.nx * self.ny,
                dtype=add_features_to_map.dtype,
                device=add_features_to_map.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = add_features_to_map[batch_mask, :]
            pillars = pillars.t()
            add_feature[:, indices] = pillars
            batch_add_features.append(add_feature)

        batch_add_features = torch.stack(batch_add_features, 0)
        batch_add_features = batch_add_features.view(batch_size, self.num_bev_add_features * self.nz, self.ny, self.nx)
        batch_dict['add_features'] = batch_add_features
        
        return batch_dict



class PointPillarScatter_Multi_pillar_od(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.in_num_bev_features = 64
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.num_bev_add_features = 3
        self.nx, self.ny, self.nz = grid_size
        # self.nx, self.ny, self.nz = self.model_cfg.ORI_NX_NY_NZ
        
        assert self.nz == 1
        
        self.nx_multi, self.ny_multi, self.nz_multi = 1280, 100, 1
        # self.nx_multi, self.ny_multi, self.nz_multi = self.model_cfg.NX_NY_NZ
        print('self.nx, self.ny, self.nz:', self.nx, self.ny, self.nz)
        self.xy_view = SingleViewNet()
        self.cylinder_view = SingleViewNet()


    def forward(self, batch_dict, **kwargs):
        # B, N, 3

        pillar_features, coords = batch_dict['xy_feature'], batch_dict['voxel_coords']
        pillar_features_multi, coords_multi = batch_dict['cylinder_feature'], batch_dict['voxel_coords_multi']

        # spatial_feature_multi
        batch_spatial_features_multi = []
        indice_cylider = []
        batch_size = coords_multi[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.in_num_bev_features, self.nz_multi * self.nx_multi * self.ny_multi,
                dtype=pillar_features_multi.dtype,
                device=pillar_features_multi.device)

            batch_mask = coords_multi[:, 0] == batch_idx
            this_coords = coords_multi[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx_multi + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features_multi[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features_multi.append(spatial_feature)
            indice_cylider.append(indices)
        batch_spatial_features_multi = torch.stack(batch_spatial_features_multi, 0)
        batch_spatial_features_multi = batch_spatial_features_multi.view(batch_size, self.in_num_bev_features*self.nz_multi, self.ny_multi, self.nx_multi)

        # spatial features
        batch_spatial_features = []
        indice_xy = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.in_num_bev_features, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
            indice_xy.append(indices)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.in_num_bev_features * self.nz, self.ny, self.nx)
        
        batch_xy_features = self.xy_view(batch_spatial_features)
        batch_cylinder_features = self.cylinder_view(batch_spatial_features_multi)
        
        _,_,H,W = batch_xy_features.size()
        # H,W = 320, 320
        
        batch_cylinder_features = torch.nn.functional.interpolate(batch_cylinder_features, size=(H, W))
        # batch_xy_features = torch.nn.functional.interpolate(batch_xy_features, size=(H, W))
        
        batch_spatial_features = torch.cat([batch_xy_features, batch_cylinder_features], dim=1)
        
        batch_dict['spatial_features'] = batch_spatial_features
        
        # add_features
        add_features_to_map, coords = batch_dict['add_features_to_map'], batch_dict['voxel_coords']
        
        batch_add_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            add_feature = torch.zeros(
                self.num_bev_add_features,
                self.nz * self.nx * self.ny,
                dtype=add_features_to_map.dtype,
                device=add_features_to_map.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = add_features_to_map[batch_mask, :]
            pillars = pillars.t()
            add_feature[:, indices] = pillars
            batch_add_features.append(add_feature)

        batch_add_features = torch.stack(batch_add_features, 0)
        batch_add_features = batch_add_features.view(batch_size, self.num_bev_add_features * self.nz, self.ny, self.nx)
        # batch_add_features = torch.nn.functional.interpolate(batch_add_features, size=(H, W))
        
        batch_dict['add_features'] = batch_add_features
        
        return batch_dict

class SingleViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = BasicBlock(64, 64, 1)
        self.res2 = BasicBlock(64, 64, 2)
        self.res3 = BasicBlock(64, 64, 2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, (2, 2), stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 64, (4, 4), stride=4, bias=False)
        self.conv = nn.Conv2d(64*3, 64, (3, 3), stride=1, padding=1)
    
    def forward(self, pillar_feature):
        batch_size = pillar_feature.size()[0]
        x1 = self.res1(pillar_feature)
        x2 = self.res2(pillar_feature)
        x3 = self.res3(x2)
        x2 = self.deconv2(x2)
        x3 = self.deconv3(x3)
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv(x_concat)
        return x_out

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