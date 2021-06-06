from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import numpy as np
import os
import os.path as osp
import json
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from PIL import Image
from IPython import embed
from copy import deepcopy
import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch


SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test'}
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']
LidarName = 'LIDAR_TOP'


def get_dense_depth_ann(ann_token: str, timestamp, points, nusc):
    """
    points: np.array of shape [3, N]
    """
    points = deepcopy(points)
    ann_data = nusc.get('sample_annotation', ann_token)
    
    visibilities = ['1', '2', '3', '4']
    #visibilities=['2', '3', '4']
    if not (ann_data['visibility_token'] in visibilities):
        return None, None, None
    # 'sample_token', 'instance_token', 'translation', 'size', 'rotation', 'prev': '', 'next', 'num_lidar_pts'
    #box = Box(ann_data['translation'], ann_data['size'], Quaternion(ann_data['rotation']),
    #               name=ann_data['category_name'], token=ann_data['token'])
    box = nusc.get_box(ann_token)
    # collect points

    is_valid_points = is_points_in_box(box, deepcopy(points))
    valid_points = points[:, is_valid_points]
    print('number valid points: {} - gt:{}'.format(valid_points.shape[1], ann_data['num_lidar_pts']))

    box_center = box.center
    valid_points = np.concatenate([valid_points, box_center[:, np.newaxis]], axis=-1)
    # get speed twice

    # avg speed:  pts_x, pts_y, pts_dep, pts_sf
    dense_points = dense_points_from_box(box)

    valid_points = np.concatenate([valid_points, dense_points], axis=-1)
    

    

    

    return valid_points, box
    

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

def points_from_box(box, points):
    """
    box.orientation.rotation_matrix, 
    box.center 
    points: np.array: [3, N]
    """
    points = np.dot(box.orientation.rotation_matrix, points)
    x, y, z = box.center

    points[0, :] = points[0, :] + x
    points[1, :] = points[1, :] + y
    points[2, :] = points[2, :] + z

    return points


def dense_points_from_box(box):
    """
    box.orientation.rotation_matrix, 
    box.center 
    """
    
    w, l, h = box.wlh
    x_lim, y_lim, z_lim = l / 2, w / 2, h / 2

    x_lim, y_lim, z_lim = x_lim * 0.75, y_lim * 0.75, z_lim * 0.75

    x_range = np.arange(-x_lim, x_lim, 0.2)
    y_range = np.arange(-y_lim, y_lim, 0.2)
    z_range = np.arange(-z_lim, z_lim, 0.2)

    x_cord, y_cord, z_cord = np.meshgrid(x_range, y_range, z_range)

    x_cord = x_cord.reshape(-1,)
    y_cord = y_cord.reshape(-1,)
    z_cord = z_cord.reshape(-1,)

    points = np.stack([x_cord, y_cord, z_cord], axis=0)

    points = np.dot(box.orientation.rotation_matrix, points)
    x, y, z = box.center
    points[0, :] = points[0, :] + x
    points[1, :] = points[1, :] + y
    points[2, :] = points[2, :] + z

    return points


def is_points_in_box(box, points):
    '''
    points: np.array of shape [3, N]
    '''
    
    points = points_to_box(box, points)
    points = np.abs(points)

    w, l, h = box.wlh
    
    # 这一步有点不确定了
    x_lim, y_lim, z_lim = l / 2, w / 2, h / 2

    in_box_x = points[0, :] < x_lim
    in_box_y = points[1, :] < y_lim
    in_box_z = points[2, :] < z_lim

    in_box = np.logical_and(np.logical_and(in_box_x, in_box_y), in_box_z)

    return in_box


def parse_sample(sample_token, nusc):

    ret_dict = {}
    sample_data = nusc.get('sample', sample_token)
    timestamp = sample_data['timestamp']

    # First Step: Get lidar points in global frame cord!
    lidar_token = sample_data['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    pcl_path = osp.join(nusc.dataroot, lidar_data['filename'])
    pc = LidarPointCloud.from_file(pcl_path)

    # lidar point in point sensor frame

    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # First Step finished! pc in global frame

    for cam_name in CamNames:
        cam_token = sample_data['data'][cam_name]
        _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)

        all_points = []
        for box in boxes:
            ann_token = box.token
            points, _ = get_dense_depth_ann(ann_token, timestamp, deepcopy(pc.points[:3, ]), nusc)
            if points is not None:
                all_points.append(points)

        if len(all_points) > 0:
            points = np.concatenate(all_points, axis=-1)
        else:
            # set meta[''] = None!
            ret_dict[cam_name] = None
            continue
        # transform points to ego pose; change sf's orentiation

        # change from global to ego pose
        points = points - np.array(poserecord['translation'])[:, np.newaxis]
        points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

        # change from ego to camera
        cam_data = nusc.get('sample_data', cam_token)
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # transform points for ego pose to camera pose
        cam_points = deepcopy(points) - np.array(cam_cs['translation'])[:, np.newaxis]
        cam_points = np.dot(Quaternion(cam_cs['rotation']).rotation_matrix.T, cam_points)

        ret_points = cam_points
        ret_dict[cam_name] = ret_points

    return ret_dict


def visualize_sample(sample_token, nusc):
    ret_dict = parse_sample(sample_token, nusc)
    sample_data = nusc.get('sample', sample_token)

    for cam_name in CamNames:
        if ret_dict[cam_name] is None:
            # set xxx to None
            continue
        cam_token = sample_data['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        im = Image.open(osp.join(nusc.dataroot, cam_data['filename']))
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        ret_points = ret_dict[cam_name]
        cam_points = ret_points
        
        # use camera intrinsics to project points into image plane. 
        cam_points = view_points(cam_points, np.array(cam_cs['camera_intrinsic']), normalize=True)

        # filter out points which do not show in this camera
        mask = np.ones(cam_points.shape[1], dtype=bool)
        
        mask = np.logical_and(mask, cam_points[0, :] > 1)
        mask = np.logical_and(mask, cam_points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, cam_points[1, :] > 1)
        mask = np.logical_and(mask, cam_points[1, :] < im.size[1] - 1)
        #print('Mask num', cam_points.shape[1], mask.sum())
        cam_points = cam_points[:, mask]
        img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR) 
        for i in range(cam_points.shape[1]):
            cv2.circle(img, (int(cam_points[0, i]), int(cam_points[1, i])), 4, [0,0,255], -1)
 
        
        _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        for box in boxes:
            c = nusc.colormap[box.name]
            box.render_cv2(img, view=np.array(cam_cs['camera_intrinsic']), normalize=True, colors=(c, c, c))
        
        im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        im.save(os.path.join('/home/zhangty/tmp/visual_sf', cam_name+'.png'))


def generate_dense_depth_for_one_sample(sample_token, nusc, save_dir, nusc_exp):
    ret_dict = parse_sample(sample_token, nusc)
    sample_data = nusc.get('sample', sample_token)
    lidar_token = sample_data['data'][LidarName]   

    ret_list = []
    for cam_name in CamNames:
        
        data_info = {}
        cam_token = sample_data['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        filename = cam_data['filename']

        data_info['img_path'] = filename
        data_info['sample_token'] = sample_token
        data_info['lidar_token'] = lidar_token
        data_info['cam_token'] = cam_token

        if ret_dict[cam_name] is None:
            points, coloring, im = nusc_exp.map_pointcloud_to_image(lidar_token, cam_token)

            float_x_cords = points[0]  # < 1600
            float_y_cords = points[1]  # < 900
            float_depth = coloring  # 

            points_with_depth = np.stack([float_x_cords, float_y_cords, float_depth], axis=0)

            cam_points = points_with_depth
            cam_points = cam_points.transpose()

            img_name = osp.split(filename)[-1].split('.')[0]
            np.save(osp.join(save_dir, img_name), cam_points) # will add .npy postfix automaticlly

            data_info['points_path'] = osp.join(save_dir, img_name)

            ret_list.append(data_info)
            continue
 
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        ret_points = ret_dict[cam_name]
        cam_points = ret_points
        cam_points = view_points(cam_points, np.array(cam_cs['camera_intrinsic']), normalize=True)

        # filter out points which do not show in this camera
        mask = np.ones(cam_points.shape[1], dtype=bool)
        
        mask = np.logical_and(mask, cam_points[0, :] > 1)
        mask = np.logical_and(mask, cam_points[0, :] < 1599)
        mask = np.logical_and(mask, cam_points[1, :] > 1)
        mask = np.logical_and(mask, cam_points[1, :] < 899)
        cam_points = cam_points[:, mask]

        points, coloring, im = nusc_exp.map_pointcloud_to_image(lidar_token, cam_token)

        float_x_cords = points[0] # < 1600
        float_y_cords = points[1] # < 900
        float_depth = coloring # 

        points_with_depth = np.stack([float_x_cords, float_y_cords, float_depth], axis=0)

        cam_points = np.concatenate([cam_points, points_with_depth], axis=1)
        cam_points = cam_points.transpose()

        img_name = osp.split(filename)[-1].split('.')[0]
        np.save(osp.join(save_dir, img_name), cam_points) # will add .npy postfix automaticlly

        data_info['points_path'] = osp.join(save_dir, img_name)

        ret_list.append(data_info)
        
    return ret_list

def _main():
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    nusc_exp = NuScenesExplorer(nusc)
    samples = nusc.sample

    save_dir = '/public/MARS/datasets/nuScenes-SF/dense-objects-depth'
    points_save_dir = '/public/MARS/datasets/nuScenes-SF/dense-objects-depth/data'
    meta = []
    for sample in tqdm(samples):
        sample_token = sample['token']
        ret_list = generate_dense_depth_for_one_sample(sample_token, nusc, points_save_dir, nusc_exp)
        meta = meta + ret_list

    meta_file_path = os.path.join(save_dir, 'meta.json')
    with open(meta_file_path, 'w') as f:
        json.dump(meta, f)


def _test_visual(idx=0):
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)

    samples = nusc.sample

    sample = samples[idx]
    sample_token = sample['token']
    #tp_parse_sample(sample_token, nusc)
    visualize_sample(sample_token, nusc)
    embed()
    # visualize_sample(samples[3]['token'], nusc)


if __name__ == '__main__':
    _main()
    
    #_test_visual()
