
# import rospy
# import ros_numpy
# import numpy as np
# import copy
# import json
# import os
# import sys
# import torch
# import time 
# import argparse
# import glob
# from pathlib import Path
# import message_filters
# from std_msgs.msg import Header
# from pyquaternion import Quaternion
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2, PointField
# from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray


# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.utils import common_utils


# class DemoDataset(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
#         """
#         Args:
#             root_path:
#             dataset_cfg:
#             class_names:
#             training:
#             logger:
#         """
#         super().__init__(
#             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
#         )
#         self.root_path = root_path
#         self.ext = ext
#         data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

#         data_file_list.sort()
#         self.sample_file_list = data_file_list

#     def __len__(self):
#         return len(self.sample_file_list)

#     def __getitem__(self, index):
#         if self.ext == '.bin':
#             points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
#         elif self.ext == '.npy':
#             points = np.load(self.sample_file_list[index])
#         else:
#             raise NotImplementedError

#         input_dict = {
#             'points': points,
#             'frame_id': index,
#         }

#         data_dict = self.prepare_data(data_dict=input_dict)
#         return data_dict

# def yaw2quaternion(yaw: float) -> Quaternion:
#     return Quaternion(axis=[0,0,1], radians=yaw)

# def get_annotations_indices(types, thresh, label_preds, scores):
#     indexs = []
#     annotation_indices = []
#     for i in range(label_preds.shape[0]):
#         if label_preds[i] == types:
#             indexs.append(i)
#     for index in indexs:
#         if scores[index] >= thresh:
#             annotation_indices.append(index)
#     return annotation_indices  


# def remove_low_score_nu(image_anno, thresh):
#     img_filtered_annotations = {}
#     label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
#     scores_ = image_anno["scores"].detach().cpu().numpy()
    
#     car_indices =                  get_annotations_indices(0, 0.45, label_preds_, scores_)
#     truck_indices =                get_annotations_indices(1, 0.45, label_preds_, scores_)
#     construction_vehicle_indices = get_annotations_indices(2, 0.45, label_preds_, scores_)
#     bus_indices =                  get_annotations_indices(3, 0.35, label_preds_, scores_)
#     trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
#     barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
#     motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
#     bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
#     pedestrain_indices =           get_annotations_indices(8, 0.10, label_preds_, scores_)
#     traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
#     for key in image_anno.keys():
#         if key == 'metadata':
#             continue
#         img_filtered_annotations[key] = (
#             image_anno[key][car_indices +
#                             pedestrain_indices + 
#                             bicycle_indices +
#                             bus_indices +
#                             construction_vehicle_indices +
#                             traffic_cone_indices +
#                             trailer_indices +
#                             barrier_indices +
#                             truck_indices
#                             ])

#     return img_filtered_annotations


# class Processor_ROS:
#     def __init__(self, config_path, model_path):
#         self.points = None
#         self.config_path = config_path
#         self.model_path = model_path
#         self.device = None
#         self.net = None
#         self.voxel_generator = None
#         self.inputs = None
        
#     def initialize(self):
#         self.read_config()
        
#     def read_config(self):
#         config_path = self.config_path
#         cfg_from_yaml_file(self.config_path, cfg)
#         self.logger = common_utils.create_logger()
#         self.demo_dataset = DemoDataset(
#             dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#             root_path=Path("/home/muzi2045/Documents/project/OpenPCDet/data/kitti/velodyne_enhanced/000001.bin"),
#             ext='.bin')
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
#         self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
#         self.net = self.net.to(self.device).eval()

#     def run(self, points):
#         t_t = time.time()
#         print(f"input points shape: {points.shape}")
#         num_features = 4        
#         self.points = points.reshape([-1, num_features])

#         input_dict = {
#             'points': self.points,
#             'frame_id': 0,
#         }

#         data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
#         data_dict = self.demo_dataset.collate_batch([data_dict])
#         load_data_to_gpu(data_dict)

#         torch.cuda.synchronize()
#         t = time.time()

#         pred_dicts, _ = self.net.forward(data_dict)
        
#         torch.cuda.synchronize()
#         print(f" pvrcnn inference cost time: {time.time() - t}")

#         boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
#         scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
#         types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

#         # print(f" pred boxes: { boxes_lidar }")
#         # print(f" pred_scores: { scores }")
#         # print(f" pred_labels: { types }")

#         return scores, boxes_lidar, types

# def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
#     '''
#     '''
#     if remove_nans:
#         mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
#         cloud_array = cloud_array[mask]

#     points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
#     points[...,0] = cloud_array['x']
#     points[...,1] = cloud_array['y']
#     points[...,2] = cloud_array['z']
#     return points

# def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
#     '''
#     Create a sensor_msgs.PointCloud2 from an array of points.
#     '''
#     msg = PointCloud2()
#     if stamp:
#         msg.header.stamp = stamp
#     if frame_id:
#         msg.header.frame_id = frame_id
#     msg.height = 1
#     msg.width = points_sum.shape[0]
#     msg.fields = [
#         PointField('x', 0, PointField.FLOAT32, 1),
#         PointField('y', 4, PointField.FLOAT32, 1),
#         PointField('z', 8, PointField.FLOAT32, 1)
#         # PointField('i', 12, PointField.FLOAT32, 1)
#         ]
#     msg.is_bigendian = False
#     msg.point_step = 12
#     msg.row_step = points_sum.shape[0]
#     msg.is_dense = int(np.isfinite(points_sum).all())
#     msg.data = np.asarray(points_sum, np.float32).tostring()
#     return msg

# def rslidar_callback(lidar_msg1, lidar_msg2):
#     t_t = time.time()
#     arr_bbox = BoundingBoxArray()

#     msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
#     np_p = get_xyz_points(msg_cloud, True)
#     print("  ")
#     scores, dt_box_lidar, types = proc_1.run(np_p)

#     if scores.size != 0:
#         for i in range(scores.size):
#             bbox = BoundingBox()
#             bbox.header.frame_id = msg.header.frame_id
#             bbox.header.stamp = rospy.Time.now()
#             q = yaw2quaternion(float(dt_box_lidar[i][6]))
#             bbox.pose.orientation.x = q[1]
#             bbox.pose.orientation.y = q[2]
#             bbox.pose.orientation.z = q[3]
#             bbox.pose.orientation.w = q[0]           
#             bbox.pose.position.x = float(dt_box_lidar[i][0])
#             bbox.pose.position.y = float(dt_box_lidar[i][1])
#             bbox.pose.position.z = float(dt_box_lidar[i][2])
#             bbox.dimensions.x = float(dt_box_lidar[i][3])
#             bbox.dimensions.y = float(dt_box_lidar[i][4])
#             bbox.dimensions.z = float(dt_box_lidar[i][5])
#             bbox.value = scores[i]
#             bbox.label = int(types[i])
#             arr_bbox.boxes.append(bbox)
#     print("total callback time: ", time.time() - t_t)
#     arr_bbox.header.frame_id = msg.header.frame_id
#     arr_bbox.header.stamp = msg.header.stamp
#     if len(arr_bbox.boxes) is not 0:
#         pub_arr_bbox.publish(arr_bbox)
#         arr_bbox.boxes = []
#     else:
#         arr_bbox.boxes = []
#         pub_arr_bbox.publish(arr_bbox)
   
# if __name__ == "__main__":

#     global proc
#     ## PVRCNN
#     config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml'
#     model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/pv_rcnn_8369.pth'

#     proc_1 = Processor_ROS(config_path, model_path)
    
#     proc_1.initialize()
    
#     rospy.init_node('centerpoint_ros_node')
#     sub_lidar_topic = [ "/velodyne_enhanced_points", 
#                         "/top/rslidar_points",
#                         "/points_raw", 
#                         "/lidar_protector/merged_cloud", 
#                         "/merged_cloud",
#                         "/lidar_top", 
#                         "/roi_pclouds",
#                         "/livox/lidar/time_sync",
#                         "/SimOneSM_PointCloud_0"]
    
#     # sub_ = rospy.Subscriber(sub_lidar_topic[7], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
#     lidar_sub1 = message_filters.Subscriber('/livox/lidar_1HDDH1200100801/time_sync', PointCloud2)
#     lidar_sub2 = message_filters.Subscriber('/livox/lidar_3WEDH7600103381/time_sync', PointCloud2)

#     ts = message_filters.ApproximateTimeSynchronizer([lidar_sub1, lidar_sub2], 1, 0.1)
#     ts.registerCallback(rslidar_callback)
#     pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)

#     print("[+] PVRCNN ros_node has started!")    
#     rospy.spin()

# import rospy
# import ros_numpy
import numpy as np
import os
import sys
import torch
import time 
import glob
from pathlib import Path

# from std_msgs.msg import Header
# from pyquaternion import Quaternion
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2, PointField
# from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import cv2
from pcdet.utils import calibration_kitti
os.environ["CUDA_VISIBLE_DEVICES"]='0'

pointcloud_list_ground = []
times = []
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.txt':
            column_idx = [0,1,2,5,6]
            points = np.loadtxt(self.sample_file_list[index], dtype=np.float32).reshape(-1, 8)[:, column_idx]
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        print(points.shape())

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["pred_labels"].detach().cpu().numpy()
    scores_ = image_anno["pred_scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.45, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.45, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.45, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.35, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.10, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations
import argparse
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/yanqiao/Radar_OpenPCDet/tools/cfgs/Radar_Thermal_models/RTDF_RCNN_CA.yaml', help='specify the config for training')
    parser.add_argument('--model_path', type=str, default='/home/newdisk/yanqiao/output/Radar_Thermal_models/RTDF_RCNN_CA/default/ckpt/checkpoint_epoch_100.pth', help='specify the config for training')



    args = parser.parse_args()
    return args

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.number = 0
        self.times = []


        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            # root_path=Path("/home/newdisk/yanqiao/dataset/view-of-Delft/view_of_delft_PUBLIC/radar/training/velodyne_enhanced/00001.bin"),
            # ext='.bin')
            root_path=Path("/home/newdisk/yanqiao/dataset/Radar_Thermal/training/velodyne_enhanced/000000.txt"),
            ext='.txt')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('trainable params:',pytorch_total_params / (10**6))
        self.net = self.net.eval()
        
        

        

    def run(self):
        for i in range(1):
            file_idx =  "%06d.txt" % i
            file = "/home/newdisk/yanqiao/dataset/Radar_Thermal/training/velodyne_enhanced/" + "%06d.txt" % i
            # print(file)
            dir_list = os.listdir("/home/newdisk/yanqiao/dataset/Radar_Thermal/training/velodyne_enhanced")
            if file_idx not in dir_list:
                continue 
            # points  = np.fromfile(str(file), dtype=np.float32).reshape(-1, 7)
            column_idx = [0,1,2,5,6]
            points = np.loadtxt(str(file), dtype=np.float32).reshape(-1, 8)[:, column_idx]
            img_file = "/home/newdisk/yanqiao/dataset/Radar_Thermal/training/image_2/" + "%06d.png" % i
            img = cv2.imread(img_file)
            calib_file = "/home/newdisk/yanqiao/dataset/Radar_Thermal/training/calib/" + "%06d.txt" % i
            calib = calibration_kitti.Calibration(calib_file)
            # print(f"input points shape: {points.shape}")
            num_features = 5      
            self.points = points.reshape([-1, num_features])
            self.points = points
            self.imgs = img
            self.calib = calib

            # input_dict = {
            #     'points': self.points,
            #     'frame_id': 0,
            # }
            input_dict = {
                'points': self.points,
                'frame_id': 0,
                'images': self.imgs,
                'calib': self.calib,
            }

            data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            torch.cuda.synchronize()
            t = time.time()

            pred_dicts, _ = self.net.forward(data_dict)
            
            torch.cuda.synchronize()
            self.times.append(time.time() - t)
            print(f" inference time: {time.time() - t}")
            print(f" inference mean fps: {1/np.min(self.times)}")
   
if __name__ == "__main__":

    global proc
    config_path = '/home/yanqiao/Radar_OpenPCDet/tools/cfgs/Radar_Thermal_models/RTDF_RCNN_CA.yaml'
    model_path = '/home/newdisk/yanqiao/output/Radar_Thermal_models/RTDF_RCNN_CA/default/ckpt/checkpoint_epoch_100.pth'

    # config_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pv_rcnn/default/pv_rcnn.yaml'
    # model_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pv_rcnn/default/ckpt/checkpoint_epoch_40.pth'

    # config_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all/default/pointpillar_all.yaml'
    # model_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all/default/ckpt/checkpoint_epoch_80.pth'
    
    # config_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all_multiview_pillar/default/pointpillar_all_multiview_pillar.yaml'
    # model_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all_multiview_pillar/default/ckpt/checkpoint_epoch_80.pth'

    # config_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all_multiview_pillar_addfeatures/default/pointpillar_all_multiview_pillar_addfeatures.yaml'
    # model_path = '/home/yanqiao/Radar_OpenPCDet/output/VoD_models/pointpillar_all_multiview_pillar_addfeatures/default/ckpt/checkpoint_epoch_80.pth'
    
    args = parse_config()
    
    proc_1 = Processor_ROS(args.cfg_file, args.model_path)
    
    proc_1.initialize()
    proc_1.run()
    
