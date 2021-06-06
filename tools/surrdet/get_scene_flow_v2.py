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

SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test'}
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']

def get_sf_ann(ann_token:str, timestamp, points, nusc):
    """
    points: np.array of shape [3, N]
    """
    points = deepcopy(points)
    ann_data = nusc.get('sample_annotation', ann_token)
    
    visibilities=['1', '2', '3', '4']
    #visibilities=['2', '3', '4']
    if not (ann_data['visibility_token'] in visibilities):
        return None, None, None
    # 'sample_token', 'instance_token', 'translation', 'size', 'rotation', 'prev': '', 'next', 'num_lidar_pts'
    #box = Box(ann_data['translation'], ann_data['size'], Quaternion(ann_data['rotation']),
    #               name=ann_data['category_name'], token=ann_data['token'])
    box = nusc.get_box(ann_token)
    # collect points
    next_ann_token = ann_data['next']
    prev_ann_token = ann_data['prev']

    has_next = next_ann_token != ''
    has_prev = prev_ann_token != ''

    if (not has_next) and (not has_prev):
        return None, None, box

    num_ref = int(has_next) + int(has_prev)

    is_valid_points = is_points_in_box(box, deepcopy(points))
    valid_points = points[:, is_valid_points]
    print('number valid points: {} - gt:{}'.format(valid_points.shape[1], ann_data['num_lidar_pts']))
    #assert valid_points.shape[1] == ann_data['num_lidar_pts'], 'num of lidar pts not right! {} {}'.format(valid_points.shape[1], ann_data['num_lidar_pts'])

    box_center = box.center
    #valid_points = np.concatenate([valid_points, box_center[:, np.newaxis]], axis=-1)
    # get speed twice

    # avg speed:  pts_x, pts_y, pts_dep, pts_sf
    valid_points_to_box = points_to_box(box, deepcopy(valid_points))

    sf = 0
    if next_ann_token is not '':
        next_points, next_time = points_to_ann_cord(next_ann_token, deepcopy(valid_points_to_box), nusc)
        sf_from_next = (next_points - valid_points) / (next_time*1e-6 - timestamp*1e-6)
        sf = sf + sf_from_next

    if prev_ann_token is not '':
        prev_points, prev_time = points_to_ann_cord(prev_ann_token, deepcopy(valid_points_to_box), nusc)
        sf_from_prev = (prev_points - valid_points) / (prev_time*1e-6 - timestamp*1e-6)
        sf = sf + sf_from_prev

    #embed()  
    sf = sf / num_ref

    return valid_points, sf, box


def points_to_ann_cord(ann_token, points, nusc):
    ann_data = nusc.get('sample_annotation', ann_token)
    # 'sample_token', 'instance_token', 'translation', 'size', 'rotation', 'prev': '', 'next', 'num_lidar_pts'
    #box = Box(ann_data['translation'], ann_data['size'], Quaternion(ann_data['rotation']),
    #               name=ann_data['category_name'], token=ann_data['token'])
    box = nusc.get_box(ann_token)
    next_sample = nusc.get('sample', ann_data['sample_token'])

    timestamp = next_sample['timestamp']
    return points_from_box(box, points), timestamp

    

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
        all_sf = []
        for box in boxes:
            ann_token = box.token
            points, sf, _ = get_sf_ann(ann_token, timestamp, deepcopy(pc.points[:3, ]), nusc)
            if points is not None:
                all_points.append(points)
                all_sf.append(sf)

        if len(all_points) > 0:
            points = np.concatenate(all_points, axis=-1)
            sf = np.concatenate(all_sf, axis=-1)
        else:
            # set meta[''] = None!
            ret_dict[cam_name] = None
            continue
        # transform points to ego pose; change sf's orentiation

        # change from global to ego pose
        points = points - np.array(poserecord['translation'])[:, np.newaxis]
        points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

        sf = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, sf)

        # change from ego to camera
        cam_data = nusc.get('sample_data', cam_token)
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # transform points for ego pose to camera pose
        cam_points = deepcopy(points) - np.array(cam_cs['translation'])[:, np.newaxis]
        cam_points = np.dot(Quaternion(cam_cs['rotation']).rotation_matrix.T, cam_points)
        cam_sf = np.dot(Quaternion(cam_cs['rotation']).rotation_matrix.T, sf)

        ret_points = np.concatenate([cam_points, cam_sf], axis=0)
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
        cam_points, cam_sf = ret_points[:3, ], ret_points[3:, ]
        
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
        cam_sf = cam_sf[:, mask]
        img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR) 
        prev_point = np.array([0,0])
        for i in range(cam_points.shape[1]):
            cur_cord = np.array([int(cam_points[0, i]), int(cam_points[1, i])])
            cv2.circle(img, (int(cam_points[0, i]), int(cam_points[1, i])), 4, [0,0,255], -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            sf_str = '{:.3f} - {:.3f} - {:.3f}'.format(cam_sf[0, i], cam_sf[1, i], cam_sf[2, i])

            
            direct = np.array([cam_sf[0, i], cam_sf[1, i]])
            #direct = direct / (np.sum(np.abs(direct)) + 1e-4) * 80
            direct = direct  * 100

            end_cord = cur_cord + direct.astype(np.int)

            print(sf_str, direct, end_cord)
            
            if np.sum(np.abs(cur_cord - prev_point)) > 300:
                cv2.arrowedLine(img, cur_cord, end_cord, [0, 255, 0], 5) 
                prev_point = cur_cord
        
        _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        for box in boxes:
            c = nusc.colormap[box.name]
            box.render_cv2(img, view=np.array(cam_cs['camera_intrinsic']), normalize=True, colors=(c, c, c))
        
        im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        im.save(os.path.join('/home/zhangty/tmp/visual_sf', cam_name+'.png'))


def visualize_sample_bev(sample_token, nusc):
    ret_dict = parse_sample(sample_token, nusc)
    sample_data = nusc.get('sample', sample_token)

    point_list = []
    motion_point_list = []

    for cam_name in CamNames:
        if ret_dict[cam_name] is None:
            # set xxx to None
            continue
        cam_token = sample_data['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        #im = Image.open(osp.join(nusc.dataroot, cam_data['filename']))
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        ret_points = ret_dict[cam_name]
        cam_points, cam_sf = ret_points[:3, ], ret_points[3:, ]
        # transform points for ego pose to camera pose
        cam_points = np.dot(Quaternion(cam_cs['rotation']).rotation_matrix, deepcopy(cam_points))
        cam_points = cam_points + np.array(cam_cs['translation'])[:, np.newaxis]
        
        cam_sf = np.dot(Quaternion(cam_cs['rotation']).rotation_matrix, cam_sf)

        points = np.concatenate([cam_points, cam_sf], axis=0)

        motion_point_list.append(points)
    
    points = np.concatenate(motion_point_list, axis=1)

    #plt.scatter(pc[:, 0], pc[:, 1])

        


def generate_sf_for_one_sample(sample_token, nusc, save_dir, meta=None):
    ret_dict = parse_sample(sample_token, nusc)
    sample_data = nusc.get('sample', sample_token)

    for cam_name in CamNames:
        
        ret = {}
        cam_token = sample_data['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        filename = cam_data['filename']

        ret['img_path'] = filename
        if ret_dict[cam_name] is None:
            
            ret['points_path'] = None
            meta[cam_token] = ret
            continue
 
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        ret_points = ret_dict[cam_name]
        cam_points, cam_sf = ret_points[:3, ], ret_points[3:, ]
        cam_points = view_points(cam_points, np.array(cam_cs['camera_intrinsic']), normalize=True)

        # filter out points which do not show in this camera
        mask = np.ones(cam_points.shape[1], dtype=bool)
        
        mask = np.logical_and(mask, cam_points[0, :] > 1)
        mask = np.logical_and(mask, cam_points[0, :] < 1599)
        mask = np.logical_and(mask, cam_points[1, :] > 1)
        mask = np.logical_and(mask, cam_points[1, :] < 899)
        cam_points = cam_points[:, mask]
        cam_sf = cam_sf[:, mask]

        save_points = np.concatenate([cam_points, cam_sf], axis=0)

        img_name = osp.split(filename)[-1].split('.')[0]
        np.save(osp.join(save_dir, img_name), save_points) # will add .npy postfix automaticlly
        save_path = osp.join(img_name +'.npy')

        ret['points_path'] = save_path
        meta[cam_token] = ret
    
    return meta

def _main():
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)

    samples = nusc.sample

    save_dir = '/public/MARS/datasets/nuScenes-SF/trainval-camera'
    meta = {}
    for sample in tqdm(samples):
        sample_token = sample['token']
        generate_sf_for_one_sample(sample_token, nusc, save_dir, meta)

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
