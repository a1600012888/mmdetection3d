from mmcv import visualization
from numpy.lib.arraysetops import isin
import torch
from torch._C import DeviceObjType
import torch.nn as nn
import numpy as np

from mmdet.models import DETECTORS
from .depth_net import PackNetSlim01
from .sf_net import SFNet
from .lidar_loss.temporal_spatial_loss import warp_ref_image_temporal, \
    calc_scene_flow_consistency_loss, warp_ref_image_spatial, calc_depth_consistency_loss, \
    calc_invdepth_consistency_loss, calc_stereo_rgb_loss

from .utils import get_depth_metrics, get_motion_metrics, remap_invdepth_color,\
    get_smooth_loss, group_smoothness,  sparsity_loss, get_motion_metrics, \
    flow2rgb, convert_res_sf_to_sf, sf_sparse_loss, get_bidirection_motion_meterics

from .loss import PhotoMetricLoss

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
                scale_depth_for_temp=True,
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
        self.photometric_loss = PhotoMetricLoss(w_l1=0.15, w_census=0.85, reduce='min')
        self.scale_depth = scale_depth
        self.scale_depth_for_temp = scale_depth_for_temp
        # < 0 means no supervised depth loss
        self.depth_supervision_ratio = depth_supervision_ratio
        self.depth_smoothing = depth_smoothing
        self.motion_smoothing = motion_smoothing
        self.motion_sparse = motion_sparse
        self.sf_consis = sf_consis
        self.depth_consis = depth_consis
        self.rgb_consis = rgb_consis
        self.stereo_rgb_consis = stereo_rgb_consis
        self.loss_decay = loss_decay

        self.depth_net.eval()
        #check = torch.load('/public/MARS/models/surrdet/depth-model/packnet_lr2e-4_decay1e-1_16epoch-weights.pth', map_location='cpu')['state_dict']
        check = torch.load('/public/MARS/surrdet/tyz/depth-net.pth', map_location='cpu')['state_dict']
        self.depth_net.load_state_dict(check)
        self.depth_net.eval()


    def forward_train(self, data, **kwargs):
        # prev: 0-5; now: 6-11; next: 12:17
        # img0, img1, img2, ... img17
        img_keys = ['img{}'.format(i) for i in range(18)]
        self.depth_net.eval()

        now_depth_keys = ['depth_map{}'.format(i) for i in range(6, 12)]
        now_sf_next_keys = ['sf_next_map{}'.format(i) for i in range(6, 12)]
        now_sf_prev_keys = ['sf_prev_map{}'.format(i) for i in range(6, 12)]

        # Imgs!
        # shape [18, N, 3, H, W]
        imgs = [data[tmp_key] for tmp_key in img_keys]

        prev_imgs, now_imgs, next_imgs = imgs[0:6], imgs[6:12], imgs[12:]
        # shape [N*6, 3, H, W]
        prev_imgs = torch.cat(prev_imgs, dim=0)
        now_imgs = torch.cat(now_imgs, dim=0)
        next_imgs = torch.cat(next_imgs, dim=0)

        now2next_time = data['now2next_time'].view(-1,1,1,1)
        now2prev_time = data['now2prev_time'].view(-1,1,1,1)

        #from IPython import embed
        #embed()
        # Depths GT!
        # shape [N*6, H, W]
        now_depth = torch.cat([data[tmp_key] for tmp_key in now_depth_keys], dim=0)
        # shape [N*6, 1, H, W]
        now_depth = now_depth.unsqueeze(dim=1)

        # SceneFlow GT!
        # shape [N*6, H, W, 4]
        now_sf_next = torch.cat([data[tmp_key] for tmp_key in now_sf_next_keys], dim=0)
        now_sf_prev = torch.cat([data[tmp_key] for tmp_key in now_sf_prev_keys], dim=0) # div by negative time
        # shape [N*6, 4, H, W]
        #from IPython import embed
        #embed()
        now_sf_next = now_sf_next.permute(0, 3, 1, 2)
        now_sf_prev = now_sf_prev.permute(0, 3, 1, 2)

        # Depth Pred!
        # shape [N*6, 1, H, W]

        all_imgs = torch.cat([prev_imgs, now_imgs, next_imgs], dim=0)

        with torch.no_grad():
            all_disp_preds = self.depth_net.get_pred(all_imgs)

        # SceneFlow Pred!
        # shape [N*6, 3, H, W]
        # now2next_sf_pred = self.sf_net(now_imgs, next_imgs)
        # now2prev_sf_pred = self.sf_net(now_imgs, prev_imgs)

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

        disp_raw = all_disp_preds[0].detach()
        #from IPython import embed
        #embed()
        # note: following loss can only be computed with batch size as one
        # now_inv_depth: [N*6, 1, H, W]
        prev_disp, now_disp, next_disp = torch.split(disp_raw, 6, dim=0)
        _, _, Hp, Wp = now_disp.shape
        if Hp != H:
            now_inv_depth = nn.functional.interpolate(now_disp, size=[H, W])
            # print(Hp, Wp, H, W)
            prev_disp = nn.functional.interpolate(prev_disp, size=[H, W])
            next_disp = nn.functional.interpolate(next_disp, size=[H, W])
        else:
            now_inv_depth = now_disp

        depth_pred = 1.0 / torch.clamp(now_inv_depth, min=1e-6)
        prev_depth_pred = 1.0 / torch.clamp(prev_disp, min=1e-6)
        next_depth_pred = 1.0 / torch.clamp(next_disp, min=1e-6)
        if self.scale_depth_for_temp:
            depth_scale = torch.median(now_depth[mask]) / (torch.median(depth_pred[mask]) + 1e-4)
            depth_scale = depth_scale.detach()
        else:
            depth_scale = 1.0

        # scaling the depth prediction!

        prev_imgs_depth = torch.cat([prev_imgs, prev_depth_pred], dim=1)
        next_imgs_depth = torch.cat([next_imgs, next_depth_pred], dim=1)

        now_imgs_depth = torch.cat([now_imgs, depth_pred.detach()], dim=1)

        now2next_inp = torch.cat([now_imgs_depth.detach(), next_imgs_depth], dim=1)
        now2prev_inp = torch.cat([now_imgs_depth.detach(), prev_imgs_depth], dim=1)
        now2next_sf_pred = self.sf_net(now2next_inp)
        now2prev_sf_pred = self.sf_net(now2prev_inp)

        rec_prev_imgs = warp_ref_image_temporal(now_inv_depth.detach(), prev_imgs,
                                        now_cam_intrin, prev_cam_intrin,
                                        now_cam_pose, prev_cam_pose,
                                        now2prev_sf_pred)

        rec_next_imgs = warp_ref_image_temporal(now_inv_depth.detach(), next_imgs,
                                        now_cam_intrin, next_cam_intrin,
                                        now_cam_pose, next_cam_pose,
                                        now2next_sf_pred)

        # [N, 1, H, W]
        temp_rec_loss = self.photometric_loss(now_imgs, [rec_next_imgs, rec_prev_imgs, next_imgs, prev_imgs])

        motion_smoothing_loss = get_smooth_loss(now2prev_sf_pred, now_imgs) + \
                                get_smooth_loss(now2next_sf_pred, now_imgs)

        motion_sparse_loss, res_now2next_sf_pred = sf_sparse_loss(now_cam_intrin, next_cam_intrin,
                                        now_cam_pose, next_cam_pose, depth_pred, now2next_sf_pred)
        motion_sparse_loss_, res_now2prev_sf_pred = sf_sparse_loss(now_cam_intrin, prev_cam_intrin,
                                                    now_cam_pose, prev_cam_pose, depth_pred, now2prev_sf_pred)
        motion_sparse_loss = motion_sparse_loss + motion_sparse_loss_
        # need to change the sparse loss(use gt depth?)
        # motion_sparse_loss = sparsity_loss(now2prev_sf_pred) + \
        #                         sparsity_loss(now2next_sf_pred)

        loss = self.rgb_consis * temp_rec_loss

        loss = loss + self.motion_smoothing * motion_smoothing_loss

        loss = loss + self.motion_sparse * motion_sparse_loss

        with torch.no_grad():
            metrics = get_depth_metrics(depth_pred, now_depth, mask, scale=self.scale_depth)
            # abs_diff, abs_rel, sq_rel, rmse, rmse_log
            metrics = [m.item() for m in metrics]
            # depth_scale = median(gt_depth) / median(pred_depth)
            abs_diff, abs_rel, sq_rel, rmse, rmse_log, depth_scale = metrics

            # add visualization map

            std = torch.tensor([255.0, 255.0, 255.0]).cuda().view(1, -1, 1, 1)
            mean = torch.tensor([0.0, 0.0, 0.0]).cuda().view(1, -1, 1, 1)
            img = now_imgs * std + mean
            img = img / 255.0
            prev_imgs = prev_imgs * std + mean
            prev_imgs = prev_imgs / 255.0
            next_imgs = next_imgs * std + mean
            next_imgs = next_imgs / 255.0

            rec_prev_imgs = rec_prev_imgs * std + mean
            rec_prev_imgs = rec_prev_imgs / 255.0

            rec_next_imgs = rec_next_imgs * std + mean
            rec_next_imgs = rec_next_imgs / 255.0
            # scale the depth for visualization
            inv_depth_pred_img0 = (now_inv_depth[0]).clamp(min=1e-5)

            cmap_inv_depth_pred_img0 = remap_invdepth_color(inv_depth_pred_img0)

            visualization_map = {
                'inv_depth_pred': cmap_inv_depth_pred_img0.transpose(2,0,1),
                'img': img[0].detach().cpu().numpy(),
                'prev_img': prev_imgs[0].detach().cpu().numpy(),
                'next_img': next_imgs[0].detach().cpu().numpy(),
                'rec_from_prev': rec_prev_imgs[0].detach().cpu().numpy(),
                'rec_from_next': rec_next_imgs[0].detach().cpu().numpy(),
            }

            log_vars = {'loss': loss.item(),
                        'temp_rec_loss': temp_rec_loss.item(),
                        'motion_smoothing': motion_smoothing_loss.item(),
                        'motion_sparse': motion_sparse_loss.item(),
                        'sparsity': sparsity.item(),
                        'abs_diff': abs_diff, 'abs_rel': abs_rel,
                        'avg_time_diff': now2next_time.mean().item(), 
                        }

        with torch.no_grad():
            #fused_motion = (now2next_sf_pred - now2prev_sf_pred) / 2
            motion_metrics = get_bidirection_motion_meterics(now2next_sf_pred, now2prev_sf_pred, 
                                            now_sf_next, now_sf_prev, now2next_time, now2prev_time, 
                                            now_cam_intrin, now_cam_pose, next_cam_intrin, next_cam_pose, 
                                            prev_cam_intrin, prev_cam_pose, 
                                            now_depth, self.scale_depth)
            #motion_metrics = [m.item() for m in motion_metrics]
            epe, epe_rel, motion_scale, thres_3, thres_5, thres_10 = motion_metrics
            log_vars['epe'] = epe
            log_vars['epe_rel'] = epe_rel
            log_vars['motion_scale'] = motion_scale
            log_vars['thres_0.03'] = thres_3
            log_vars['thres_0.05'] = thres_5
            log_vars['thres_0.10'] = thres_10

            visual_motion = now2next_sf_pred[0]
            visual_motion = flow2rgb(visual_motion.detach().cpu().numpy().transpose(1,2,0)).transpose(2, 0, 1)

            res_visual_motion = res_now2next_sf_pred[0]
            res_visual_motion = flow2rgb(res_visual_motion.detach().cpu().numpy().transpose(1,2,0)).transpose(2, 0, 1)
            cat_a, cat_b = visualization_map['img'], visualization_map['inv_depth_pred']
            pred_visual = np.concatenate([cat_a, cat_b], axis=-1)
            pred_visual_ = np.concatenate([visual_motion, res_visual_motion], axis=-1)

            pred_visual = np.concatenate([pred_visual, pred_visual_], axis=-2)

            time_warp_visual = np.concatenate([cat_a,
                                visualization_map['rec_from_prev'],
                                visualization_map['rec_from_next']], axis=-1)

            image_prev_now_next = np.concatenate([visualization_map['prev_img'],
                                        cat_a,
                                        visualization_map['prev_img'],], axis=-1)

            final_visualization_map = {'img-depth-motion': pred_visual,
                                        'time_warp': time_warp_visual,
                                        'image_time': image_prev_now_next}

        return loss, log_vars, final_visualization_map

    def forward(self, return_loss=True, rescale=False, **data):
        if not return_loss:
            loss, log_vars, visualization_map = self.forward_train(data)
            ret_keys = ['abs_rel', 'loss', 'epe', 
            'epe_rel', 'thres_0.03', 'thres_0.05',
            'thres_0.10', 'temp_rec_loss']

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


    def eval_forward(self, data):
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

        now_inv_depth_preds = self.depth_net.get_pred(now_imgs)

        camera_intrinsic = data['cam_intrinsic']
        #img_x, img_y = now_imgs.size(-2), now_imgs.size(-1)
        img_x, img_y = now_imgs.size(-1), now_imgs.size(-2)
        #print(now_imgs.shape)
        scale_x = img_x / 1600
        scale_y = img_y / 900
        camera_intrinsic = scale_intrinsics(camera_intrinsic, scale_x, scale_y)
        now_cam_intrin = camera_intrinsic[:, 6:12, ...].view(-1, 3, 3)
        # [N, 18, 4, 4]
        cam_pose = data['cam_pose']
        # [N, 6, 4, 4] => [N*6, 4, 4]
        now_cam_pose = cam_pose[:, 6:12, ...].view(-1, 4, 4)

        outputs = {'inv_depth_preds': now_inv_depth_preds,
                    'cam_intrinsic': now_cam_intrin,
                    'cam_pose': now_cam_pose,
                    'imgs': now_imgs}
        return outputs
    
    def preprocess_forward(self, data):
        img_keys = ['img{}'.format(i) for i in range(18)]
        self.depth_net.eval()

        imgs = [data[tmp_key] for tmp_key in img_keys]

        prev_imgs, now_imgs, next_imgs = imgs[0:6], imgs[6:12], imgs[12:]
        # shape [N*6, 3, H, W]
        prev_imgs = torch.cat(prev_imgs, dim=0)
        now_imgs = torch.cat(now_imgs, dim=0)
        next_imgs = torch.cat(next_imgs, dim=0)


        # Depth Pred!
        # shape [N*6, 1, H, W]

        all_imgs = torch.cat([prev_imgs, now_imgs, next_imgs], dim=0)

        with torch.no_grad():
            all_disp_preds = self.depth_net.get_pred(all_imgs)

        B, _, H, W = now_imgs.shape

        disp_raw = all_disp_preds[0].detach()
        #from IPython import embed
        #embed()
        # note: following loss can only be computed with batch size as one
        # now_inv_depth: [N*6, 1, H, W]
        #prev_disp, now_disp, next_disp = torch.split(disp_raw, 6, dim=0)
        
        return disp_raw