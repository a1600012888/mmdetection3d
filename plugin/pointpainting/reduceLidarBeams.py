import mmcv
import torch
import math
import time
import numpy as np
from torchvision import utils as vutils
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class ReduceLiDARBeams(object):
    
    def __init__(self, reduce_beams_to=32):
        self.reduce_beams_to = reduce_beams_to

    def __call__(self, results):
        points = results['points']
        pts = points.tensor
        print(pts.size())
        # 
        pts_3d = pts[:, :3]
        radius = math.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
        sine_phi = pts[:, 2] / radius
        phi = torch.asin(sine_phi)
        sine_theta = pts[:, 1] / (radius * torch.cos(phi))

        im = torch.zeros(2000, 2000, 3)
        x_img = int(sine_phi + 1) * 1000
        y_img = int(sine_theta + 1) * 1000
        im[x_img.long(), y_img.long(), :] = 1

        im = im.permute(2, 0, 1)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        print(timestamp)
        vutils.save_image(im, 'rangeview/' + timestamp + '.jpg')