import torch
import torch.nn as nn

from mmdet.models import DETECTORS
from .depth_net import PackNetSlim01
from .sf_net import SFNet
from .lidar_loss.temporal_spatial_loss import warp_ref_image_temporal, PhotometricLoss, \
    calc_scene_flow_consistency_loss, warp_ref_image_spatial, calc_depth_consistency_loss


def scale_intrinsics(K, x_scale, y_scale):
    """
    Scale intrinsics given x_scale and y_scale factors
    x_scale = x_tgt_size / x_prev_size

    """
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

@DETECTORS.register_module()
class SpatialTempNet(nn.Module):

    def __init__(self, depth_net_cfg, sf_net_cfg=None, **kwargs):

        super(SpatialTempNet, self).__init__()
        self.depth_net = PackNetSlim01(**depth_net_cfg)
        self.sf_net = SFNet()
        self.photometric_loss = PhotometricLoss(alpha=0.1, clip_loss=0.5, C1=1e-4, C2=9e-4)

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
        #from IPython import embed
        #embed()
        now_sf = now_sf.permute(0, 3, 1, 2)

        # Depth Pred!
        # shape [N*6, 1, H, W]

        now_inv_depth = self.depth_net.get_pred(now_imgs)

        # SceneFlow Pred!
        # shape [N*6, 3, H, W]
        now2next_sf_pred = self.sf_net(now_imgs, next_imgs)
        now2prev_sf_pred = self.sf_net(now_imgs, prev_imgs)

        # [N, 18, 3, 3]
        camera_intrinsic = data['cam_intrinsic']
        img_x, img_y = now_imgs.size(-2), now_imgs.size(-1)
        scale_x = img_x / 1600
        scale_y = img_y / 900
        camera_intrinsic = scale_intrinsics(camera_intrinsic, scale_x, scale_y)
        prev_cam_intrin = camera_intrinsic[:, 0:6, ...].view(-1, 3, 3)
        now_cam_intrin = camera_intrinsic[:, 6:12, ...].view(-1, 3, 3)
        next_cam_intrin = camera_intrinsic[:, 12:18, ...].view(-1, 3, 3)
        # [N, 18, 4, 4]
        cam_pose = data['cam_pose']
        prev_cam_pose = cam_pose[:, 0:6, ...].view(-1, 4, 4)
        now_cam_pose = cam_pose[:, 6:12, ...].view(-1, 4, 4)
        next_cam_pose = cam_pose[:, 12:18, ...].view(-1, 4, 4)

        #from IPython import embed
        #embed()
        rec_prev_imgs = warp_ref_image_temporal(now_inv_depth, prev_imgs,
                                        now_cam_intrin, prev_cam_intrin,
                                        now_cam_pose, prev_cam_pose,
                                        now2prev_sf_pred)
        rec_next_imgs = warp_ref_image_temporal(now_inv_depth, next_imgs,
                                        now_cam_intrin, next_cam_intrin,
                                        now_cam_pose, next_cam_pose,
                                        now2next_sf_pred)

        
        temp_rec_losses = self.photometric_loss([rec_prev_imgs], [prev_imgs]) + \
                        self.photometric_loss([rec_next_imgs], [next_imgs])

        #from IPython import embed
        #embed()
        consis_sf_loss = calc_scene_flow_consistency_loss([now2prev_sf_pred, now2next_sf_pred], now_cam_intrin, now_cam_pose, now_inv_depth)
        consis_inv_dep_loss, valid_mask_ratio = calc_depth_consistency_loss(now_inv_depth, now_cam_intrin, now_cam_pose)
        temp_rec_loss = temp_rec_losses[0].mean(dim=0).mean()

        loss = temp_rec_loss + consis_sf_loss + consis_inv_dep_loss
        log_vars = {'loss': loss.item(),
                    'temp_rec_loss': temp_rec_loss.item(), 
                    'inv_depth_consis_loss': consis_inv_dep_loss.item(), 
                    'sceneflow_consis_loss': consis_sf_loss.item(), 
                    'valid_mask_ratio': valid_mask_ratio, 
                    }

        return loss, log_vars

    def train_step(self, data, optimzier):

        loss, log_vars = self(data)
        
        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        outputs = {'loss':loss, 'log_vars':log_vars,
                    'num_samples':data['img1'].size(0)}
        #print('output', outputs)
        return outputs

    def val_step(self, data, optimizer):

        return self.train_step(self, data, optimizer)
