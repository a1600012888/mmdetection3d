import argparse
import time
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
import os
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def project_radar_to_camera(points, pc_range, lidar2img, img_shape, img, id=0):
    '''
    points: [N, 3]
    pc_range:
    lidar2img: [4, 4]
    img_shape: [2]
    '''
    points[..., 0:1] = points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1:2] = points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2:3] = points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    lidar2img = np.asarray(lidar2img)

    lidar2img = points.new_tensor(lidar2img)
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    # [N, 4, 1]
    points = points.unsqueeze(dim=-1)
    # [N, 4]
    points_cam = torch.matmul(lidar2img, points).squeeze(-1)

    eps = 1e-5
    mask = (points_cam[..., 2:3] > eps)
    points_cam = points_cam[..., 0:2] / torch.maximum(
        points_cam[..., 2:3], torch.ones_like(points_cam[..., 2:3])*eps)
    
    mask = (mask & (points_cam[..., 0:1] < img_shape[1])
                 & (points_cam[..., 0:1] > 0)
                 & (points_cam[..., 1:2] < img_shape[0])
                 & (points_cam[..., 1:2] > 0))
    
    
    mask = mask.squeeze(dim=-1)
    points_cam = points_cam[mask]

    points_cam = points_cam[..., 0:2]

    img = img.detach().cpu().numpy()
    mean = [103.530, 116.280, 123.675]
    mean = np.array(mean).reshape(3, 1, 1)
    
    img = img + mean
    #img = img.astype(np.int)
    img = img.transpose(1,2,0).astype(np.uint8).copy()
    points_cam = points_cam.detach().cpu().numpy().astype(np.int)

    for point in points_cam:
        
        img = cv2.circle(img, point, 10, (0, 255, 0), -1)

    cv2.imwrite('./visual/sample-{}.jpg'.format(id), img)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):
        if i % 10 != 0:
            continue
        img = data['img'][0].data[0][0][0]
        img_shape = data['img_metas'][0].data[0][0]['img_shape'][0]
        lidar2img = data['img_metas'][0].data[0][0]['lidar2img'][0]
        radar_points = data['radar'][0].data[0][..., 0:3]
        #print(img.shape, img_shape, lidar2img.shape, radar_points.shape)

        project_radar_to_camera(points=radar_points, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                lidar2img=lidar2img, img_shape=img_shape, img=img, id=i)
        
        if i > 1000:
            break

if __name__ == '__main__':
    main()
