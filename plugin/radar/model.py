from mmcv import visualization
from numpy.lib.arraysetops import isin
import torch
from torch._C import DeviceObjType
import torch.nn as nn
import numpy as np

from mmdet.models import DETECTORS
from .depth_net import PackNetSlim01
from .sf_net import SFNet
from .lidar_loss.temporal_spatial_loss import warp_ref_image_temporal, PhotometricLoss, \
    calc_scene_flow_consistency_loss, warp_ref_image_spatial, calc_depth_consistency_loss, \
    calc_invdepth_consistency_loss, calc_stereo_rgb_loss

from .utils import get_depth_metrics, get_motion_metrics, remap_invdepth_color,\
    get_smooth_loss, group_smoothness,  sparsity_loss, get_motion_metrics, \
    flow2rgb


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
    _left_swap = [i for i in range(-1,5)]
    _right_swap = [i % 6 for i in range(1,7)]

    def __init__(self, depth_net_cfg, sf_net_cfg=None,
                scale_depth=False, 
                depth_supervision_ratio=1.0,
                depth_smoothing=1e-2, 
                motion_smoothing=1e-3,
                motion_sparse=1e-3, 
                sf_consis=1.0, 
                depth_consis=1.0, 
                rgb_consis=1.0, 
                stereo_rgb_consis=1.0, 
                loss_decay=1.0, 
                **kwargs):

        super(SpatialTempNet, self).__init__()
        self.depth_net = PackNetSlim01(**depth_net_cfg)
        self.sf_net = SFNet()
        self.photometric_loss = PhotometricLoss(alpha=0.1, clip_loss=0.5, C1=1e-4, C2=9e-4)
        self.scale_depth = scale_depth
        # < 0 means no supervised depth loss
        self.depth_supervision_ratio= depth_supervision_ratio
        self.depth_smoothing = depth_smoothing
        self.motion_smoothing = motion_smoothing
        self.motion_sparse = motion_sparse
        self.sf_consis = sf_consis
        self.depth_consis = depth_consis
        self.rgb_consis = rgb_consis
        self.stereo_rgb_consis = stereo_rgb_consis
        self.loss_decay = loss_decay
        

    def forward_train(self, data, **kwargs):
        # prev: 0-5; now: 6-11; next: 12:17
        # img0, img1, img2, ... img17
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

        now_inv_depth_preds = self.depth_net.get_pred(now_imgs)

        # SceneFlow Pred!
        # shape [N*6, 3, H, W]
        now2next_sf_pred = self.sf_net(now_imgs, next_imgs)
        now2prev_sf_pred = self.sf_net(now_imgs, prev_imgs)

        # [N, 18, 3, 3]
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
        # [N, 18, 4, 4]
        cam_pose = data['cam_pose']

        # [N, 6, 4, 4] => [N*6, 4, 4]
        prev_cam_pose = cam_pose[:, 0:6, ...].view(-1, 4, 4)
        now_cam_pose = cam_pose[:, 6:12, ...].view(-1, 4, 4)
        next_cam_pose = cam_pose[:, 12:18, ...].view(-1, 4, 4)

        mask = now_depth > 0
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)

        B, _, H, W = now_imgs.shape
        loss_list = []
        log_vars_list = []
        for pred_idx, now_inv_depth_raw in enumerate(now_inv_depth_preds):
            #from IPython import embed
            #embed()
            # note: following loss can only be computed with batch size as one
            # now_inv_depth: [N*6, 1, H, W]
            _, _, Hp, Wp = now_inv_depth_raw.shape
            if Hp != H:
                now_inv_depth = nn.functional.interpolate(now_inv_depth_raw, size=[H, W])
            else:
                now_inv_depth = now_inv_depth_raw

            rec_prev_imgs = warp_ref_image_temporal(now_inv_depth, prev_imgs,
                                            now_cam_intrin, prev_cam_intrin,
                                            now_cam_pose, prev_cam_pose,
                                            now2prev_sf_pred)
            rec_next_imgs = warp_ref_image_temporal(now_inv_depth, next_imgs,
                                            now_cam_intrin, next_cam_intrin,
                                            now_cam_pose, next_cam_pose,
                                            now2next_sf_pred)
            
            #rec_left_imgs = warp_ref_image_temporal(now_inv_depth, now_imgs)

            # [N, 1, H, W]
            temp_rec_loss = self.photometric_loss([rec_prev_imgs], [now_imgs])[0] + \
                            self.photometric_loss([rec_next_imgs], [now_imgs])[0]
            temp_rec_loss = temp_rec_loss.mean()

            stereo_left_loss, left_rec_imgs = calc_stereo_rgb_loss(now_inv_depth, now_imgs[self._left_swap, ...], now_imgs, 
                                                now_cam_intrin, now_cam_intrin[self._left_swap, ...], 
                                                now_cam_pose, now_cam_pose[self._left_swap, ], )
            stereo_right_loss, right_rec_imgs = calc_stereo_rgb_loss(now_inv_depth, now_imgs[self._right_swap, ...], now_imgs, 
                                                now_cam_intrin, now_cam_intrin[self._right_swap, ...], 
                                                now_cam_pose, now_cam_pose[self._right_swap, ], )

            stereo_rgb_loss = stereo_left_loss + stereo_right_loss
            #from IPython import embed
            #embed()
            depth_pred = 1.0 / torch.clamp(now_inv_depth, 1e-6)

            consis_sf_loss, sf_valid_mask_ratio = calc_scene_flow_consistency_loss([now2prev_sf_pred, now2next_sf_pred], now_cam_intrin, now_cam_pose, now_inv_depth)
            #consis_inv_dep_loss, valid_mask_ratio = calc_invdepth_consistency_loss(now_inv_depth, now_cam_intrin, now_cam_pose)
            consis_dep_loss, valid_mask_ratio = calc_depth_consistency_loss(depth_pred, now_cam_intrin, now_cam_pose)
            consis_dep_loss = consis_dep_loss
                

            depth_smoothing_loss = get_smooth_loss(depth_pred, now_imgs)
            #motion_smoothing_loss = get_smooth_loss(now2prev_sf_pred, now_imgs) + get_smooth_loss(now2next_sf_pred, now_imgs)
            motion_smoothing_loss = group_smoothness(now2prev_sf_pred) + group_smoothness(now2next_sf_pred) 
            motion_sparse_loss = sparsity_loss(now2prev_sf_pred) + sparsity_loss(now2next_sf_pred)

            loss = self.rgb_consis * temp_rec_loss
            loss = self.stereo_rgb_consis * stereo_rgb_loss
            loss = loss + self.sf_consis * consis_sf_loss
            loss = loss + self.depth_consis * consis_dep_loss
            loss = loss + self.depth_smoothing * depth_smoothing_loss 
            loss = loss + self.motion_smoothing * motion_smoothing_loss
            loss = loss + self.motion_sparse * motion_sparse_loss
            if self.depth_supervision_ratio > 0:
                
                # TODO: smooth L1;  imbalance L1
                depth_sup_loss = torch.abs((now_depth - depth_pred)) * mask
                depth_sup_loss = torch.sum(depth_sup_loss) / torch.sum(mask)

                loss = loss + self.depth_supervision_ratio * depth_sup_loss

            with torch.no_grad():
                metrics = get_depth_metrics(depth_pred, now_depth, mask, scale=self.scale_depth)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]
                # depth_scale = median(gt_depth) / median(pred_depth)
                abs_diff, abs_rel, sq_rel, rmse, rmse_log, depth_scale = metrics

                # add visualization map
                if pred_idx == 0:
                    #std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
                    #mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
                    #img = now_imgs * std + mean
                    #img = img / 255.0
                    img = now_imgs

                    inv_depth_pred_img0 = now_inv_depth[0].clamp(min=1e-5)

                    cmap_inv_depth_pred_img0 = remap_invdepth_color(inv_depth_pred_img0)
                    

                    visualization_map = {
                        'inv_depth_pred': cmap_inv_depth_pred_img0.transpose(2,0,1),
                        'img': img[0].detach().cpu().numpy(),
                        'rec_from_prev': rec_prev_imgs[0].detach().cpu().numpy(),
                        'rec_from_next': rec_next_imgs[0].detach().cpu().numpy(),
                        'rec_from_left': left_rec_imgs[0].detach().cpu().numpy(),
                        'rec_from_right': right_rec_imgs[0].detach().cpu().numpy(),
                    }

                log_vars = {'loss': loss.item(),
                            'temp_rec_loss': temp_rec_loss.item(),
                            'stereo_rgb_loss': stereo_rgb_loss.item(), 
                            'depth_consis_loss': consis_dep_loss.item(),
                            'sceneflow_consis_loss': consis_sf_loss.item(),
                            'depth_smoothing': depth_smoothing_loss.item(), 
                            'motion_smoothing': motion_smoothing_loss.item(), 
                            'motion_sparse_loss': motion_sparse_loss.item(), 
                            'valid_mask_ratio': valid_mask_ratio,
                            'sparsity': sparsity.item(),
                            'abs_diff': abs_diff, 'abs_rel': abs_rel,
                            'sq_rel': sq_rel, 'rmse': rmse,
                            'rmse_log': rmse_log, 'depth_scale': depth_scale
                            }
            #print(log_vars, '---', pred_idx)
            if self.depth_supervision_ratio > 0:
                log_vars['depth_sup_loss'] = depth_sup_loss.item()

            loss_list.append(loss)
            log_vars_list.append(log_vars)
        
        #print(log_vars)
        loss_ratio = 1.0 
        final_loss = 0
        for temp_loss in loss_list:
            final_loss = final_loss + temp_loss * loss_ratio
            loss_ratio = loss_ratio * self.loss_decay
        
        final_log_vars = {}
        for log_vars in log_vars_list:
            for name, value in log_vars.items():
                if name not in final_log_vars:
                    final_log_vars[name] = [value]
                else:
                    final_log_vars[name].append(value)
        
        for name, value_list in final_log_vars.items():
            final_log_vars[name] = sum(value_list) / len(value_list)
            #print('mean-', len(value_list))

        with torch.no_grad():
            fused_motion = (now2next_sf_pred - now2prev_sf_pred) / 2
            motion_metrics = get_motion_metrics(fused_motion, 
                                                now_sf, scale=self.scale_depth)
            motion_metrics = [m.item() for m in motion_metrics]
            epe, epe_rel, motion_scale = motion_metrics
            final_log_vars['epe'] = epe
            final_log_vars['epe_rel'] = epe_rel
            final_log_vars['motion_scale'] = motion_scale

            visual_motion = fused_motion[0]
            visual_motion = flow2rgb(visual_motion.detach().cpu().numpy().transpose(1,2,0)).transpose(2, 0, 1)
            cat_a, cat_b = visualization_map['img'], visualization_map['inv_depth_pred']
            pred_visual = np.concatenate([cat_a, cat_b, visual_motion], axis=-1)

            time_warp_visual = np.concatenate([cat_a, 
                                visualization_map['rec_from_prev'], 
                                visualization_map['rec_from_next']], axis=-1)
            spatial_warp_visual = np.concatenate([cat_a, 
                                visualization_map['rec_from_right'], 
                                visualization_map['rec_from_left']], axis=-1)
                                
            final_visualization_map = {'img-depth-motion': pred_visual, 
                                        'time_warp': time_warp_visual, 
                                        'spatial_warp': spatial_warp_visual}

        return final_loss, final_log_vars, final_visualization_map

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
