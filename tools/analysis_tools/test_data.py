import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
#from mmdet.core import wrap_fp16_model
from tools.misc.fuse_conv_bn import fuse_module
import os

from plugin.radar-v3.utils import get_depth_metrics, get_motion_metrics, remap_invdepth_color,\
    get_smooth_loss, group_smoothness,  sparsity_loss, get_motion_metrics, \
    flow2rgb, convert_res_sf_to_sf, sf_sparse_loss

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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin') & cfg.plugin:
        import importlib
        _module_dir = os.path.dirname(args.config)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
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

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    #load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        img_keys = ['img{}'.format(i) for i in range(18)]
        now_depth_keys = ['depth_map{}'.format(i) for i in range(6, 12)]
        now_sf_keys = ['sf_map{}'.format(i) for i in range(6, 12)]

        # Imgs!
        # shape [18, N, 3, H, W]
        imgs = [data[tmp_key] for tmp_key in img_keys]
        prev_imgs, now_imgs, next_imgs = imgs[0:6], imgs[6:12], imgs[12:]
        # shape [N*6, 3, H, W]
        prev_imgs = torch.cat(prev_imgs, dim=0)
        now_imgs = torch.cat(now_imgs, dim=0)
        next_imgs = torch.cat(next_imgs, dim=0)

        now_depth = torch.cat([data[tmp_key] for tmp_key in now_depth_keys], dim=0)
        
        now_depth = now_depth.unsqueeze(dim=1)
        
        now_sf = torch.cat([data[tmp_key] for tmp_key in now_sf_keys], dim=0)
 
        now_sf = now_sf.permute(0, 3, 1, 2)

        all_imgs = torch.cat([prev_imgs, now_imgs, next_imgs], dim=0)

        
        camera_intrinsic = data['cam_intrinsic']
        #img_x, img_y = now_imgs.size(-2), now_imgs.size(-1)
        img_x, img_y = now_imgs.size(-1), now_imgs.size(-2)
        #print(now_imgs.shape)
        scale_x = img_x / 1600
        scale_y = img_y / 900
        camera_intrinsic = scale_intrinsics(camera_intrinsic, scale_x, scale_y)
        prev_cam_intrin = camera_intrinsic[:, 0:6, ...].view(-1, 3, 3)
        now_cam_intrin = camera_intrinsic[:, 6:12, ...].view(-1, 3, 3)
        next_cam_intrin = camera_intrinsic[:, 12:18, ...].view(-1, 3, 3)
        
        cam_pose = data['cam_pose']

        # [N, 6, 4, 4] => [N*6, 4, 4]
        prev_cam_pose = cam_pose[:, 0:6, ...].view(-1, 4, 4)
        now_cam_pose = cam_pose[:, 6:12, ...].view(-1, 4, 4)
        next_cam_pose = cam_pose[:, 12:18, ...].view(-1, 4, 4)


        B, _, H, W = now_imgs.shape
        
        with torch.no_grad():
            #fused_motion = (now2next_sf_pred - now2prev_sf_pred) / 2
            real_sf = convert_res_sf_to_sf(now_cam_intrin, next_cam_intrin,
                                        now_cam_pose, next_cam_pose, now_depth, now_sf/10.0)
            

if __name__ == '__main__':
    main()
