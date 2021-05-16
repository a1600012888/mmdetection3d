from mmcv import visualization
from numpy.lib.arraysetops import isin
import torch
from torch._C import DeviceObjType
import torch.nn as nn

from mmdet.models import DETECTORS
from .depth_net import PackNetSlim01
from .sf_net import SFNet
from .lidar_loss.temporal_spatial_loss import warp_ref_image_temporal, PhotometricLoss, \
    calc_scene_flow_consistency_loss, warp_ref_image_spatial, calc_depth_consistency_loss

from .utils import get_depth_metrics, remap_invdepth_color


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

    def __init__(self, depth_net_cfg, sf_net_cfg=None,
                depth_supervision_ratio=1.0,
                **kwargs):

        super(SpatialTempNet, self).__init__()
        self.depth_net = PackNetSlim01(**depth_net_cfg)
        self.sf_net = SFNet()
        self.photometric_loss = PhotometricLoss(alpha=0.1, clip_loss=0.5, C1=1e-4, C2=9e-4)
        # < 0 means no supervised depth loss
        self.depth_supervision_ratio= depth_supervision_ratio

    def forward_train(self, data, **kwargs):
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

        # [N, 6, 4, 4] => [N*6, 4, 4]
        prev_cam_pose = cam_pose[:, 0:6, ...].view(-1, 4, 4)
        now_cam_pose = cam_pose[:, 6:12, ...].view(-1, 4, 4)
        next_cam_pose = cam_pose[:, 12:18, ...].view(-1, 4, 4)

        #from IPython import embed
        #embed()
        # note: following loss can only be computed with batch size as one
        # now_inv_depth: [N*6, 1, H, W]
        rec_prev_imgs = warp_ref_image_temporal(now_inv_depth, prev_imgs,
                                        now_cam_intrin, prev_cam_intrin,
                                        now_cam_pose, prev_cam_pose,
                                        now2prev_sf_pred)
        rec_next_imgs = warp_ref_image_temporal(now_inv_depth, next_imgs,
                                        now_cam_intrin, next_cam_intrin,
                                        now_cam_pose, next_cam_pose,
                                        now2next_sf_pred)


        # [N, 1, H, W]
        temp_rec_losses = self.photometric_loss([rec_prev_imgs], [prev_imgs]) + \
                        self.photometric_loss([rec_next_imgs], [next_imgs])

        #from IPython import embed
        #embed()
        consis_sf_loss, sf_valid_mask_ratio = calc_scene_flow_consistency_loss([now2prev_sf_pred, now2next_sf_pred], now_cam_intrin, now_cam_pose, now_inv_depth)
        consis_inv_dep_loss, valid_mask_ratio = calc_depth_consistency_loss(now_inv_depth, now_cam_intrin, now_cam_pose)
        consis_inv_dep_loss = consis_inv_dep_loss
        temp_rec_loss = temp_rec_losses[0].mean(dim=0).mean()

        loss = temp_rec_loss + consis_sf_loss + consis_inv_dep_loss

        depth_pred = 1.0 / torch.clamp(now_inv_depth, 1e-6)
        mask = now_depth > 0
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)
        if self.depth_supervision_ratio > 0:

            depth_sup_loss = torch.abs((now_depth - depth_pred)) * mask
            depth_sup_loss = torch.sum(depth_sup_loss) / torch.sum(mask)

            loss = loss + depth_sup_loss * self.depth_supervision_ratio

        with torch.no_grad():
            metrics = get_depth_metrics(depth_pred, now_depth, mask, scale=False)
            # abs_diff, abs_rel, sq_rel, rmse, rmse_log
            metrics = [m.item() for m in metrics]
            abs_diff, abs_rel, sq_rel, rmse, rmse_log, depth_scale = metrics

            std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
            mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
            img = now_imgs * std + mean
            img = img / 255.0

            inv_depth_pred_img0 = now_inv_depth[0].clamp(min=1e-5)
            inv_depth_img0 = 1. / now_depth.clamp(min=1e-6)
            inv_depth_img0[now_depth <= 0.] = 0.
            inv_depth_img0 = inv_depth_img0[0]

            cmap_inv_depth_pred_img0 = remap_invdepth_color(inv_depth_pred_img0)
            cmap_inv_depth = remap_invdepth_color(inv_depth_img0)

            visualization_map = {
                'inv_depth_pred': cmap_inv_depth_pred_img0.transpose(2,0,1),
                'inv_depth_gt': cmap_inv_depth.transpose(2,0,1),
                'img': img[0],
            }

        log_vars = {'loss': loss.item(),
                    'temp_rec_loss': temp_rec_loss.item(),
                    'inv_depth_consis_loss': consis_inv_dep_loss.item(),
                    'sceneflow_consis_loss': consis_sf_loss.item(),
                    'valid_mask_ratio': valid_mask_ratio,
                    'sf_valid_mask_ratio': sf_valid_mask_ratio,
                    'sparsity': sparsity.item(),
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log, 'depth_scale': depth_scale
                    }

        if self.depth_supervision_ratio > 0:
            log_vars['depth_sup_loss'] = depth_sup_loss.item()

        #print(log_vars)
        return loss, log_vars, visualization_map

    def forward(self, return_loss=True, rescale=False, **data):
        if not return_loss:
            loss, log_vars, visualization_map = self.forward_train(data)
            ret_keys = ['abs_diff', 'abs_rel', 
                        'sq_rel', 'rmse',
                        'rmse_log','loss']

            values = [log_vars[t_key] for t_key in ret_keys]
            return [values]

        return self.forward_train(data)

    def train_step(self, data, optimzier):

        loss, log_vars, visualization_map = self.forward_train(data)

        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        outputs = {'loss':loss, 'log_vars':log_vars,
                    'num_samples':data['img1'].size(0)}

        for visual_name, visual_map in visualization_map.items():
            outputs[visual_name] = visual_map
        #print('output', outputs)
        return outputs

    def val_step(self, data, optimizer):

        return self.train_step(self, data, optimizer)
