import mmcv
import torch
import math
import time
import numpy as np
from torchvision import utils as vutils


data_root = 'data/nuscenes/samples/LIDAR_TOP/'
filename = 'n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801668197868.pcd.bin'
pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

pts = np.fromfile(data_root + filename, dtype=np.float32)
pts = pts.reshape(-1, 5)
pts = torch.from_numpy(pts)
print(pts.size())

mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
pts = pts[mask]

res = 0.1
x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
im = torch.zeros(x_max, y_max, 3)

x_img = (pts[:, 0] + pc_range[3]) / res
x_img = x_img.round().long()
y_img = (pts[:, 1] + pc_range[4]) / res
y_img = y_img.round().long()

im[x_img, y_img, :] = 1
'''
for i in [-2, 0, 2]:
    for j in [-2, 0, 2]:
        im[(x_img.long()+i).clamp(min=0, max=x_max), 
            (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
'''
im = im.permute(2, 0, 1)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
print(timestamp)
#saved_root = '/home/chenxy/mmdetection3d/'
vutils.save_image(im, 'bev/' + filename + '.jpg')