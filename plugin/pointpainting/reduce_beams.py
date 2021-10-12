import mmcv
import torch
import math
import time
import numpy as np
from torchvision import utils as vutils

data_root = 'data/nuscenes/samples/LIDAR_TOP/'
filename = 'n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801668197868.pcd.bin'

pts = np.fromfile(data_root + filename, dtype=np.float32)
pts = pts.reshape(-1, 5)
pts = torch.from_numpy(pts)
print(pts.size())
# x = r * cos(phi) * cos(theta)
# y = r * cos(phi) * sin(theta)
# z = r * sin(theta)
pts_3d = pts[:, :3]
radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
sine_phi = pts[:, 2] / radius
phi = torch.asin(sine_phi)
sine_theta = pts[:, 1] / (radius * torch.cos(phi))

im = torch.zeros(2001, 2001, 3)
x_img = (sine_phi + 1) * 1000
x_img = x_img.round()
y_img = (sine_theta + 1) * 1000
y_img = y_img.round()
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        im[(x_img.long()+i).clamp(min=0, max=2000), 
            (y_img.long()+j).clamp(min=0, max=2000), :] = 1

im = im.permute(2, 0, 1)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
print(timestamp)
#saved_root = '/home/chenxy/mmdetection3d/'
vutils.save_image(im, 'rangeview/' + filename + '.jpg')