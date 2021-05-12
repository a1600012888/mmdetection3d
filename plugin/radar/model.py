import torch
import torch.nn as nn

from mmdet.models import DETECTORS
from .depth_net import PackNetSlim01
from .sf_net import SFNet

@DETECTORS.register_module()
class SpatialTempNet(nn.Module):

    def __init__(self, depth_net_cfg, sf_net_cfg):
        super().__init__()
        self.depth_net = PackNetSlim01(**depth_net_cfg)
        self.sf_net = SFNet(**sf_net_cfg)

    def forward(self, data):
        # prev: 0-5; now: 6-11; next: 12:17
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

        # Depths GT!
        # shape [N*6, H, W]
        now_depth = torch.cat([data[tmp_key] for tmp_key in now_depth_keys], dim=0)
        # shape [N*6, 1, H, W]
        now_depth = now_depth.unsqueeze(dim=1)

        # SceneFlow GT!
        # shape [N*6, H, W, 4]
        now_sf = torch.cat([data[tmp_key] for tmp_key in now_sf_keys], dim=0)
        # shape [N*6, 4, H, W]
        now_sf = now_sf.transpose(0, 3, 1, 2)

        # Depth Pred!
        # shape [N*6, 1, H, W]
        now_depth_pred = self.depth_net(now_imgs)

        # SceneFlow Pred!
        # shape [N*6, 3, H, W]
        now2next_sf_pred = self.sf_net(now_imgs, next_imgs)
        now2prev_sf_pred = self.sf_net(now_imgs, prev_imgs)

        # [N, 18, 3, 3]
        camera_intrinsic = data['cam_intrinsic']
        # [N, 18, 4, 4]
        cam_pose = data['cam_pose']
        