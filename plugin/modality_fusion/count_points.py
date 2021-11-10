from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import numpy as np
import os
import os.path as osp
import json
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from copy import deepcopy
import torch

SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test', 'mini': 'v1.0-mini'}
Cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']
category = ['animal', 'human.pedestrian.adult',	'human.pedestrian.child',
'human.pedestrian.construction_worker','human.pedestrian.personal_mobility',
'human.pedestrian.police_officer','human.pedestrian.stroller',
'human.pedestrian.wheelchair','movable_object.barrier',
'movable_object.debris','movable_object.pushable_pullable',
'movable_object.trafficcone','static_object.bicycle_rack',
'vehicle.bicycle','vehicle.bus.bendy','vehicle.bus.rigid',
'vehicle.car','vehicle.construction','vehicle.emergency.ambulance',
'vehicle.emergency.police','vehicle.motorcycle','vehicle.trailer','vehicle.truck']
data_root = 'data/nuscenes'

top_ang = 0.1862
down_ang = -0.5353

beam_range = torch.zeros(32)
beam_range[0] = top_ang
beam_range[31] = down_ang
for i in range(1, 31):
    beam_range[i] = beam_range[i-1] - 0.023275

def points_to_box(box, points):
    '''
    box.orientation.rotation_matrix, 
    box.center 
    points: np.array: [3, N]
    '''
    x, y, z = box.center
    points[0, :] = points[0, :] - x
    points[1, :] = points[1, :] - y
    points[2, :] = points[2, :] - z

    points = np.dot(box.orientation.inverse.rotation_matrix, points)

    return points

def is_points_in_box(box, points):
    '''
    points: np.array of shape [3, N]
    '''
    
    points = points_to_box(box, points)
    points = np.abs(points)

    w, l, h = box.wlh
    
    x_lim, y_lim, z_lim = l / 2, w / 2, h / 2

    in_box_x = points[0, :] < x_lim
    in_box_y = points[1, :] < y_lim
    in_box_z = points[2, :] < z_lim

    in_box = np.logical_and(np.logical_and(in_box_x, in_box_y), in_box_z)

    return in_box

def main(split, data_path):
    beam_count = {}
    for cat in category:
        beam_count[cat] = torch.zeros(32)
    all_beam_count = torch.zeros(32)
    nusc = NuScenes(version=SPLITS[split], dataroot=data_path, verbose=True)
    samples = nusc.sample
    print(len(samples))

    for sample in samples:
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        pcl_path = osp.join(nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        points = np.fromfile(pcl_path, dtype=np.float32)
        points = points.reshape(-1, 5)
        points = torch.from_numpy(points)
        num_pts, _ = points.size()
        print(num_pts)
        radius = torch.sqrt(points[:, 0].pow(2) + points[:, 1].pow(2) + points[:, 2].pow(2))
        sine_theta = points[:, 2] / radius
        # [-pi/2, pi/2]
        theta = torch.asin(sine_theta)
        mask = []
        single_mask = (theta > (beam_range[0]-0.012))
        mask.append(single_mask)
        for id in range(1, 32):
            single_mask = (theta < (beam_range[id-1]-0.012)) * (theta > (beam_range[id]-0.012))
            mask.append(single_mask)
        # lidar point in point sensor frame
        cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        for annotation_token in sample['anns']:
            annotation_metadata =  nusc.get('sample_annotation', annotation_token)
            # print(annotation_metadata)
            box = nusc.get_box(annotation_token)
            # print(box)
            is_valid_points = is_points_in_box(box, deepcopy(pc.points[:3, ]))
            is_valid_points = torch.from_numpy(is_valid_points)
            # print(annotation_metadata['num_lidar_pts'], is_valid_points.sum())
            for id in range(32):
                valid_pts = torch.logical_and(is_valid_points, mask[id])
                beam_count[annotation_metadata['category_name']][id] += valid_pts.sum()
                all_beam_count[id] += valid_pts.sum()
    for cat in category:
        print(cat)
        for id in range(32):
            print(id, beam_count[cat][id])
    for id in range(32):
        print(id, all_beam_count[id])
        

if __name__ == '__main__':
    main('train', data_root)
    