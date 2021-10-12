from mmdet3d.core.visualizer import Visualizer
import numpy as np
import torch

data_root = 'data/nuscenes/samples/LIDAR_TOP/'
filename = 'n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801733448313.pcd.bin'

pts = np.fromfile(data_root + filename, dtype=np.float32)
pts = pts.reshape(-1, 5)
pts = torch.from_numpy(pts)
pts = pts[:, :3]
vis = Visualizer(pts)
vis.show()