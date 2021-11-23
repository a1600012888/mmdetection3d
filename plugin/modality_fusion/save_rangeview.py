import torch
import numpy as np
import math
from torchvision import utils as vutils
import time

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
    y_img = y_img.round().long()
    #print('x_img', x_img.max(), x_img.min())
    x_img = (theta + 1.2) * 1000
    x_img = x_img.round().long()
    im[x_img, y_img] = rgb[id]
    #for i in [-1, 0, 1]:
    #    for j in [-1, 0, 1]:
    #        im[(x_img.long()+i).clamp(min=0, max=2000), 
    #            (y_img.long()+j).clamp(min=0, max=2000)] = rgb[id]
    #print('angle range', id, theta.max(), theta.min(), theta.size())
    return im

def save_rangeview(pts):
    if isinstance(pts, list):
        pts = pts[0]
    print(pts.size())
    # x = r * cos(theta) * cos(phi)
    # y = r * cos(theta) * sin(phi)
    # z = r * sin(theta)

    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    # print('theta', theta.max(), theta.min())
    im = torch.zeros(2001, 2001, 3)
    #mask2 = (theta < -0.1)
    #mask = mask * mask2
    #mask = torch.logical_not(mask)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

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
        # print(beam_range[id]-0.012)
        mask2 = (theta > (beam_range[id] - 0.0120))
        im = paint_beams(im, theta[mask2], phi[mask2], id)

        mask = (theta < (beam_range[id] - 0.0120))
        sine_theta = sine_theta[mask]
        theta = theta[mask]
        phi = phi[mask]
        #print('theta', theta.max(), theta.min())
    '''

    y_img = (phi + math.pi) * 1000 / math.pi
    y_img = y_img.round().long()
    x_img = (theta + 1.2) * 1000
    x_img = x_img.round().long()
    im[x_img, y_img] = 1
    #for i in [-1, 0, 1]:
    #    for j in [-1, 0, 1]:
    #        im[(x_img.long()+i).clamp(min=0, max=2000), 
    #            (y_img.long()+j).clamp(min=0, max=2000), :] = 1

    im = im.permute(2, 0, 1)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    #saved_root = '/home/chenxy/mmdetection3d/'
    vutils.save_image(im, 'rangeview/' + 'sample_' + timestamp + '.jpg')
    
    return pts

def save_bev(pts):
    if isinstance(pts, list):
        pts = pts[0]
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
    
    #for i in [-1, 0, 1]:
    #    for j in [-1, 0, 1]:
    #        im[(x_img.long()+i).clamp(min=0, max=x_max), 
    #            (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
    
    im = im.permute(2, 0, 1)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    vutils.save_image(im, 'bev/' + timestamp + '.jpg')

def save_vismap(logodds):
    print(logodds.size())
    vismap = logodds.sum(dim=1) / 40
    print('max', vismap.max(), 'min', vismap.min())
    B, x_img, y_img = vismap.size()
    vismap = vismap.permute(1, 2, 0)
    print('vismap', vismap.size())
    im = torch.zeros(x_img, y_img, 3)
    im[..., 0:3] = torch.tensor([1, 0, 0])
    im = vismap.cpu() * im.cpu()
    im = im.permute(2, 0, 1)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    vutils.save_image(im, 'vismap/' + timestamp + '.jpg')