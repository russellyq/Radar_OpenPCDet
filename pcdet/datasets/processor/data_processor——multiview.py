from functools import partial
import math
import numpy as np
from skimage import transform
import torch

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass
from .laserscan import LaserScan

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        
        # self.laserscaner = LaserScan()

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
            # data_dict['pts_img'] = data_dict['pts_img'][mask]
            # data_dict['pts_img_grid'] = data_dict['pts_img_grid'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                # vsize_xyz=[0.32, 0.32, 5],
                # vsize_xyz=[0.08, 0.08, 5],
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
            
            # multi_view voxel
        if config.get('VOXEL_SIZE_MULTI', None)is not None:
            self.voxel_generator_multi = VoxelGeneratorWrapper(
                    vsize_xyz=[2*np.pi/2560, 0.1, 107],
                    coors_range_xyz=[0, -3, 0, np.pi, 7, 107],

                    # vsize_xyz=[2*np.pi/3200, 0.078125, 107],
                    # coors_range_xyz=[0, -3, 0, np.pi, 7, 107],
                    
                    # vsize_xyz=[2*np.pi/2000, 0.125, 107],
                    # coors_range_xyz=[0, -3, 0, np.pi, 7, 107],
                    
                    num_point_features=self.num_point_features,
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                )
        else: 
            self.voxel_generator_multi = None
            
        points = data_dict['points']
        points_cylinder = points_xyz_to_cylinder(points)
        data_dict['points_cylinder'] = points_cylinder
        

        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        """
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
        """
        
        # # multi_view voxel
        if self.voxel_generator_multi is not None:
            voxel_output_multi = self.voxel_generator_multi.generate(points_cylinder)
            voxels_multi, coordinates_multi, num_points_multi = voxel_output_multi
            if not data_dict['use_lead_xyz']:
                voxels_multi = voxels_multi[..., 3:]  # remove xyz in voxels(N, 3)
            data_dict['voxels_multi'] = voxels_multi
            data_dict['voxels_multi_xyz'] = points_cylinder[:, 0:3] / [2*np.pi/2560, 0.1, 107]
            data_dict['voxel_coords_multi'] = coordinates_multi
            data_dict['voxel_num_points_multi'] = num_points_multi
        
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)                
        data_dict['voxels'] = voxels
        data_dict['voxels_xyz'] = points[:, 0:3] / config.VOXEL_SIZE
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
                
        # print('\n voxels, ',voxels.shape)
        # print('\n voxel_coords, ', coordinates.shape)
        # print('\n voxel_num_points, ',num_points.shape)
        # print('\n voxels_multi, ',voxels_multi.shape)
        # print('\n voxel_coords_multi, ',coordinates_multi.shape)
        # print('\n voxel_num_points_multi, ',num_points_multi.shape)
        
        # self.laserscaner.set_points(points=points[:,0:3], remissions=points[:, 3], vr=points[:, 4], vr_compensate=points[:, 5])
        # proj_feature = self.laserscaner.feature_img
        # # print('\n proj_feature shape, ', proj_feature.shape)
        # data_dict['proj_feature'] = proj_feature
        # data_dict['bev'] = self.lidar_to_top(data_dict)
        
        return data_dict

    # def lidar_to_top(self, data_dict):
        
    #     TOP_X_MIN, TOP_X_MAX, TOP_Y_MIN, TOP_Y_MAX, TOP_Z_MIN, TOP_Z_MAX = 0, 51.2, -25.6, 25.6, -3, 2
    #     TOP_X_DIVISION, TOP_Y_DIVISION, TOP_Z_DIVISION = 0.16, 0.16, 5
    #     lidar = data_dict['points'] # (N, 3+C_in)
    #     idx = np.where (lidar[:,0]>TOP_X_MIN)
    #     lidar = lidar[idx]
    #     idx = np.where (lidar[:,0]<TOP_X_MAX)
    #     lidar = lidar[idx]
    #     idx = np.where (lidar[:,1]>TOP_Y_MIN)
    #     lidar = lidar[idx]
    #     idx = np.where (lidar[:,1]<TOP_Y_MAX)
    #     lidar = lidar[idx]
    #     idx = np.where (lidar[:,2]>TOP_Z_MIN)
    #     lidar = lidar[idx]
    #     idx = np.where (lidar[:,2]<TOP_Z_MAX)
    #     lidar = lidar[idx]
    #     pxs=lidar[:,0]
    #     pys=lidar[:,1]
    #     pzs=lidar[:,2]
    #     prs=lidar[:,3]
    #     pvs=lidar[:,4]
    #     pvcs=lidar[:,5]
    #     qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    #     qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    #     #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    #     qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    #     quantized = np.dstack((qxs,qys,qzs,prs,pvs,pvcs)).squeeze()
    #     X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)
    #     Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)
    #     Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION) # 0, 1
    #     height  = Xn - X0
    #     width   = Yn - Y0
    #     channel = Zn - Z0  + 2 +3 # 3+3
    #     # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    #     # top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    #     top = np.zeros(shape=(channel,height,width), dtype=np.float32)

    #     if 1:  #new method
    #         for x in range(Xn):
    #             ix  = np.where(quantized[:,0]==x)
    #             quantized_x = quantized[ix]
    #             if len(quantized_x) == 0 : continue
    #             yy = -x

    #             for y in range(Yn):
    #                 iy  = np.where(quantized_x[:,1]==y)
    #                 quantized_xy = quantized_x[iy]
    #                 count = len(quantized_xy)
    #                 if  count==0 : continue
    #                 xx = -y

    #                 # top[yy,xx,Zn+1] = min(1, np.log(count+1)/math.log(32))
    #                 top[Zn+1,yy,xx] = min(1, np.log(count+1)/math.log(32))
    #                 max_height_point = np.argmax(quantized_xy[:,2])
    #                 # top[yy,xx,Zn]=quantized_xy[max_height_point, 3]
    #                 top[Zn,yy,xx]=quantized_xy[max_height_point, 3]
    #                 max_r_point = np.argmax(quantized_xy[:,3])
    #                 # top[yy,xx,3]=quantized_xy[max_r_point, 3]
    #                 top[3,yy,xx]=quantized_xy[max_r_point, 3]
    #                 max_vr_point = np.argmax(quantized_xy[:,4])
    #                 # top[yy,xx,4]=quantized_xy[max_vr_point, 4]
    #                 top[4,yy,xx]=quantized_xy[max_vr_point, 4]
    #                 max_vrc_point = np.argmax(quantized_xy[:,5])
    #                 # top[yy,xx,5]=quantized_xy[max_vrc_point, 5]
    #                 top[5,yy,xx]=quantized_xy[max_vrc_point, 5]
                    

    #                 for z in range(Zn):
    #                     iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
    #                     quantized_xyz = quantized_xy[iz]
    #                     if len(quantized_xyz) == 0 : continue
    #                     zz = z

    #                     #height per slice
    #                     max_height = max(0,np.max(quantized_xyz[:,2])-z)
    #                     # top[yy,xx,zz]=max_height
    #                     top[zz,yy,xx]=max_height
        
    #     return top

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                # extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                try:
                    extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                except ValueError:
                    extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


def points_xyz_to_cylinder(points):
    points_xyz, points_features = points[:, 0:3], points[:, 3:]
    # points_x, points_y, points_z = torch.unbind(points_xyz, axis=-1)
    # points_rho = torch.sqrt(points_x**2 + points_y**2)
    # points_phi = torch.atan2(points_y, points_x)
    # points_cylinder = torch.stack([points_phi, points_z, points_rho], axis=-1)
    # return torch.stack([points_cylinder, points_features], axis=-1)
    points_x, points_y, points_z = np.split(points_xyz, 3, axis=-1)
    points_rho = np.sqrt(points_x**2 + points_y**2)
    points_phi = np.arctan2(points_y, points_x)
    points_cylinder = np.concatenate([points_phi, points_z, points_rho], axis=-1)
    # print(points_xyz.shape, points_cylinder.shape, points_x.shape)
    return np.concatenate([points_cylinder, points_features], axis=-1)