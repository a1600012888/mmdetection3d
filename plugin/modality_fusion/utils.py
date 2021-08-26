import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def scale_to_255(a, min, max, dtype=np.uint8):
	return ((a - min) / float(max - min) * 255).astype(dtype)

def get_pts_bev(points, reference_points, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    '''
    Args:
        points: list of [N, 3+x]
        reference_points: [num_layers, B, num_query, 3]
        pc_range [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    '''
    #print(len(points), points[0].size(), reference_points.size())
    #print(reference_points.size())
    reference_points = torch.stack((reference_points[0], reference_points[-1]), dim=0)
    # shape [2, num_query, 3]
    reference_points = reference_points[:, 0]
    points = points[0]
    mask = ((points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3]) & 
        (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] > pc_range[2]) & (points[:, 2] < pc_range[5]))
    pts = points[mask]
    points_2d = pts[:, :2]
    #print(points[i].size(), pts.size(), points_2d.size())
    points_2d[:, 0] = points_2d[:, 0] - pc_range[0]
    points_2d[:, 1] = points_2d[:, 1] - pc_range[1]

    res = 0.1 
    x_img = (points_2d[:, 0] / res).long()
    y_img = (points_2d[:, 1] / res).long()

    #pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
 
    # 创建图像数组
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(2, x_max, y_max, 3)
    im[:, x_img, y_img, :] = 1

    reference_points_2d = reference_points[:, :, :2]
    reference_points_2d[:, :, 0] = reference_points_2d[:, :, 0] * (pc_range[3] - pc_range[0])
    reference_points_2d[:, :, 1] = reference_points_2d[:, :, 1] * (pc_range[4] - pc_range[1])
    ref_x_img = (reference_points_2d[:, :, 0] / res).long()
    ref_y_img = (reference_points_2d[:, :, 1] / res).long()
    max_res = int((pc_range[3] - pc_range[0]) / res)
    #im[ref_x_img, ref_y_img, 0] = 1
    for i in range(-2, 3):
        for j in range(-2, 3):
            im[0][(ref_x_img[0]+i).clamp(0, max_res), (ref_y_img[0]+j).clamp(0, max_res), 0] = 1
            im[1][(ref_x_img[1]+i).clamp(0, max_res), (ref_y_img[1]+j).clamp(0, max_res), 0] = 1
    # shape [2, H, W, 3] --> [2, 3, H, W]
    im = im.permute(0, 3, 1 ,2)
    return im
