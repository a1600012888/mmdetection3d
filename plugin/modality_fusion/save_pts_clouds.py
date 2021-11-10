import mmcv
import torch
import math
import time
import numpy as np
from torchvision import utils as vutils
import os
import random
from nuscenes.nuscenes import NuScenes
import json
import os.path as osp
import sys
from datetime import datetime
from typing import Tuple, List, Iterable
import copy
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


sample_id = 0
pc_range = [-54, -54, -0.4, 54, 54, -0.25]


rgb = torch.zeros(100, 3)
i = 0
for r in [1, 0.66, 0.33, 0]:
    for g in [1, 0.5, 0]:
        for b in [1, 0.5, 0]:
            rgb[i][0] = r
            rgb[i][1] = b
            rgb[i][2] = g
            i = i + 1

def paint_beams(im, theta, phi, id):
    y_img = (phi + math.pi) * 1000 / math.pi
    y_img = y_img.round()
    #print('x_img', x_img.max(), x_img.min())
    x_img = (theta + 1.2) * 1000
    x_img = x_img.round()
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            im[(x_img.long()+i).clamp(min=0, max=2000), 
                (y_img.long()+j).clamp(min=0, max=2000)] = rgb[id]
    print('angle range', id, theta.max(), theta.min(), theta.size())
    return im

def map_pointcloud_to_image(nusc,
                            pts,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None,
                            show_panoptic: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])

    #pc = LidarPointCloud.from_file(pcl_path)
    #pc = LidarPointCloud(pts.T)
    pc = pts
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im

def render_pointcloud_in_image(nusc,
                               pts,
                               sample_token: str,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'LIDAR_TOP',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: List = None,
                               ax: Axes = None,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
                               lidarseg_preds_bin_path: str = None,
                               show_panoptic: bool = False):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    if show_lidarseg:
        show_panoptic = False
    sample_record = nusc.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    points, coloring, im = map_pointcloud_to_image(nusc, pts, 
                                                    pointsensor_token, camera_token,
                                                    render_intensity=render_intensity,
                                                    show_lidarseg=show_lidarseg,
                                                    filter_lidarseg_labels=filter_lidarseg_labels,
                                                    lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                    show_panoptic=show_panoptic)

    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        if lidarseg_preds_bin_path:
            fig.canvas.set_window_title(sample_token + '(predictions)')
        else:
            fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title(camera_channel)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()

def save_pts_clouds(filename):
    pts = np.fromfile(filename, dtype=np.float32)
    pts = pts.reshape(-1, 5)
    #pts = pts[:, :3]
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    #pts = pts[mask]
    
    pts = torch.from_numpy(pts)
    print(pts.size())
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    # x = r * cos(theta) * cos(phi)
    # y = r * cos(theta) * sin(phi)
    # z = r * sin(theta)
    pts_3d = pts[:, :3]

    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    print('theta', theta.max(), theta.min())
    im = torch.zeros(2001, 2001, 3)
    #mask2 = (theta < -0.1)
    #mask = mask * mask2
    #mask = torch.logical_not(mask)
    phi = torch.atan2(pts[:, 1], pts[:, 0])
    '''
    mask1 = (theta > 0.18)
    im = paint_beams(im, theta[mask1], phi[mask1], 1)

    mask = (theta < 0.18)
    sine_theta = sine_theta[mask]
    theta = theta[mask]
    phi = phi[mask]
    pts = pts[mask]
    print('theta', theta.max(), theta.min())
    '''

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang
    for i in range(1, 31):
        beam_range[i] = beam_range[i-1] - 0.023275
    #beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]
    '''
    for id in range(32):
        print(beam_range[id]-0.012)
        mask2 = (theta > (beam_range[id] - 0.0120))
        im = paint_beams(im, theta[mask2], phi[mask2], id)

        mask = (theta < (beam_range[id] - 0.0120))
        sine_theta = sine_theta[mask]
        theta = theta[mask]
        phi = phi[mask]
        #print('theta', theta.max(), theta.min())
    '''
    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    #for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
    for id in [8, 9, 10, 11]:
        beam_mask = (theta < (beam_range[id-1]-0.012)) * (theta > (beam_range[id]-0.012))
        mask = mask + beam_mask
    mask = mask.bool()
    #mask = torch.logical_not(mask)
    phi = phi[mask]
    pts = pts[mask]
    theta = theta[mask]

    y_img = (phi + math.pi) * 1000 / math.pi
    y_img = y_img.round()
    #print('x_img', x_img.max(), x_img.min())
    x_img = (theta + 1.2) * 1000
    x_img = x_img.round()
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            im[(x_img.long()+i).clamp(min=0, max=2000), 
                (y_img.long()+j).clamp(min=0, max=2000), :] = 1

    im = im.permute(2, 0, 1)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    #saved_root = '/home/chenxy/mmdetection3d/'
    vutils.save_image(im, 'rangeview/' + 'sample' + str(sample_id) + '.jpg')

    #pts = pts[mask]
    pts = pts.numpy()
    """
    pts_3d = pts[:, :3]
    saved_path = 'saved_pts_clouds/' + 'sample' + str(sample_id) + '.ply'
    np.savetxt(saved_path, pts_3d, fmt='%f %f %f')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    	    format ascii 1.0
    	    element vertex %(vert_num)d
    	    property float x
    	    property float y
    	    property float z
    	    end_header
    	    \n
    	    '''
    with open(saved_path, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(pts)))
        f.write(old)
    """
    return pts

data_root = '/home/chenxy/centerpoint/mmdetection3d/data/nuscenes/'
cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 
            'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)
my_sample = nusc.sample[sample_id]
#print(my_sample, len(nusc.sample))

LiDAR_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
pts_filename = LiDAR_data['filename']
#print(LiDAR_data)

pts = save_pts_clouds(data_root + pts_filename)
#print('ego_pts', type(ego_pts), ego_pts.shape)
pts = pts[:, :4]
pts = LidarPointCloud(pts.T)

for i in range(len(cameras)):
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', 
                camera_channel=cameras[i], out_path='rendered_imgs/' + 'sample' + str(sample_id) + cameras[i] + '.png')
    sample_record = nusc.get('sample', my_sample['token'])
    points = copy.deepcopy(pts)
    render_pointcloud_in_image(nusc, points, my_sample['token'], pointsensor_channel='LIDAR_TOP', 
                camera_channel=cameras[i], out_path='rendered_imgs/' + 'sample' + str(sample_id) + '_ego_' + cameras[i] + '.png')

