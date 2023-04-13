import numpy as np
import json
from pyquaternion import Quaternion

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = json.load(f)
    f.close()
    objects = [Object3d(line) for line in lines['objects']]
    object_to_preserve = []
    for object in objects:
        if object.cls_type == 'DontCare' or object.cls_id == -1:
            continue
        object_to_preserve.append(object)
    return object_to_preserve


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Person': 2, 'Cyclist': 3, 
                  'Bus': 1, 'Motorcyclist': 3, 'Trailer': 1, 'Truck': 1}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

def cls_type_to_kitti_type(cls_type):
    type_to_id = {'Car': 'Car', 'Person': 'Person', 'Cyclist': 'Cyclist', 
                  'Bus': 'Car', 'Motorcyclist': 'Cyclist', 'Trailer': 'Car', 'Truck': 'Car',
                  'Other Vehicle': 'DontCare'}
    if cls_type not in type_to_id.keys():
        return 'DontCare'
    return type_to_id[cls_type]

def angle_in_range(x):
    while x > np.pi:
        x = 2*np.pi - x
    while x < - np.pi:
        x = 2*np.pi + x
    return x

class Object3d(object):
    def __init__(self, line):
        center3d = line['center3d']
        classname = line['classname']
        created_by = line['created_by']
        dimension3d = line['dimension3d']
        label_certainty = line['label_certainty']
        measured_by = line['measured_by']
        object_id = line['object_id']
        occlusion = line['occlusion']
        orientation_quat = line['orientation_quat']
        score = line['score']
        
        Q = Quaternion(np.array(orientation_quat))
        yaw, _, _ = Q.yaw_pitch_roll
        
        self.truncation = 0
        self.cls_type = cls_type_to_kitti_type(classname)
        self.cls_id = cls_type_to_id(self.cls_type)
        self.occlusion = float(occlusion)  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        
        self.box2d = np.array((0,0,50,50), dtype=np.float32)
        self.h = float(dimension3d[2])
        self.w = float(dimension3d[0])
        self.l = float(dimension3d[1])
        self.loc = np.array(( - float(center3d[1]), -(float(center3d[2]) - self.h/2), float(center3d[0])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = angle_in_range( - float(yaw))
        self.alpha = angle_in_range(self.ry - np.arctan(center3d[1] / center3d[0]))
        self.score = float(score)
        self.level_str = 'None'
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        if self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
