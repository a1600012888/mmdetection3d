import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from skimage.color import lab2rgb
from .lidar_loss.geometry.camera import Camera
from .lidar_loss.geometry.pose import Pose

COS_45 = 1. / np.sqrt(2)
SIN_45 = 1. / np.sqrt(2)
TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(out_planes)
    )
class DepthPredictHead2Up(nn.Module):
    '''
    1. We use a softplus activation to generate positive depths.
        The predicted depth is no longer bounded.
    2. The network predicts depth rather than disparity, and at a single scale.
    '''

    def __init__(self, in_channels):
        super(DepthPredictHead2Up, self).__init__()

        self.up = nn.PixelShuffle(2)
        self.conv1 = conv(in_channels//4, in_channels//4, kernel_size=3)
        self.conv2 = conv(in_channels//16, in_channels//16, kernel_size=3)
        self.conv3 = conv(in_channels//64, in_channels//64, kernel_size=3)
        self.conv4 = conv(in_channels//64, in_channels//64, kernel_size=3)
        self.conv5 = conv(in_channels//64, 1, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        pred = nn.functional.softplus(x)

        return pred


def get_depth_metrics(pred, gt, mask=None, scale=False):
    """
    params:
    pred: [N,1,H,W].  torch.Tensor
    gt: [N,1,H,W].     torch.Tensor

    scale: bool. True: scale the pred depth using median(typically used for eval unsupervised depth)
    """
    if mask is not None:
        num = torch.sum(mask) # the number of non-zeros
        pred = pred[mask]
        gt = gt[mask]
    else:
        num = pred.numel()

    if scale:
        ratio = torch.median(gt) / (torch.median(pred) + 1e-4)
        pred = pred * ratio
    else:
        ratio = torch.median(gt) / (torch.median(pred) + 1e-4)
    num = num * 1.0
    diff_i = gt - pred

    abs_diff = torch.sum(torch.abs(diff_i)) / num
    abs_rel = torch.sum(torch.abs(diff_i) / gt) / num
    sq_rel = torch.sum(diff_i ** 2 / gt) / num
    rmse = torch.sqrt(torch.sum(diff_i ** 2) / num)
    rmse_log = torch.sqrt(torch.sum((torch.log(gt) -
                                        torch.log(pred)) ** 2) / num)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, torch.tensor(ratio)


def get_motion_metrics(pred, gt, scale=False):
    '''
    pred: [N, 3, H, W]
    gt: [N, 4, H, W] (last channel as mask)

    '''
    mask = gt[:, -1, :, :].type(torch.bool)

    num = torch.sum(mask)  # the number of non-zeros
    if num < 1:
        return [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    pred = torch.stack([pred[:, 0, ...][mask],
                        pred[:, 1, ...][mask],
                        pred[:, 2, ...][mask]], dim=1)
    gt = torch.stack([gt[:, 0, ...][mask],
                        gt[:, 1, ...][mask],
                        gt[:, 2, ...][mask]], dim=1)
    #print(pred.shape)

    pred_speed = (torch.sum(pred ** 2, dim=1) + 1e-6) ** 0.5
    gt_speed = (torch.sum(gt ** 2, dim=1) + 1e-6) ** 0.5
    if scale:
        ratio = torch.median(gt_speed) / (torch.median(pred_speed) + 1e-4)
        pred = pred * ratio
    else:
        ratio = torch.median(gt_speed) / (torch.median(pred_speed) + 1e-4)
    num = num * 1.0
    diff_i = gt - pred

    epe_map = (diff_i) ** 2
    #print(epe_map.shape)
    epe_map = (epe_map.sum(dim=-1, ) + 1e-6) ** 0.5
    thres_1 = torch.sum(epe_map < 0.1) / num
    thres_3 = torch.sum(epe_map < 0.3) / num
    thres_5 = torch.sum(epe_map < 0.5) / num
    #print(epe_map.shape, 'after')
    epe = torch.mean(epe_map)

    epe_rel = torch.sum(epe_map / gt_speed) / num

    return [epe, epe_rel, torch.tensor(ratio).cuda(), thres_1, thres_3, thres_5]


def get_bidirection_motion_meterics(sf_next_pred, sf_prev_pred, 
                            sf_next, sf_prev,
                            now2next_time, now2prev_time, 
                            now_cam_intrin, now_cam_pose,
                            next_cam_intrin, next_cam_pose,
                            prev_cam_intrin, prev_cam_pose,
                            now_depth_gt, scale_motion=True):

    scale_sf_next_field = sf_next[:, :3, ...] * now2next_time
    sf_next_mask = sf_next[:, [3], ...]
    scaled_sf_next = torch.cat([scale_sf_next_field, sf_next_mask], dim=1)
    real_sf_next = convert_res_sf_to_sf(now_cam_intrin, next_cam_intrin,
                                now_cam_pose, next_cam_pose, now_depth_gt, scaled_sf_next)

    scale_sf_prev_field = sf_prev[:, :3, ...] * now2prev_time
    sf_prev_mask = sf_prev[:, [3], ...]
    scaled_sf_prev = torch.cat([scale_sf_prev_field, sf_prev_mask], dim=1)
    real_sf_prev = convert_res_sf_to_sf(now_cam_intrin, prev_cam_intrin,
                                now_cam_pose, prev_cam_pose, now_depth_gt, scaled_sf_prev)

    motion_metrics_next = get_motion_metrics(sf_next_pred,  # time_delat: 0.5s / 0.1s for key/non-key
                                            real_sf_next, scale=scale_motion)
    motion_metrics_next = [m.item() for m in motion_metrics_next]

    motion_metrics_prev = get_motion_metrics(sf_prev_pred,  # time_delat: 0.5s / 0.1s for key/non-key
                                            real_sf_prev, scale=scale_motion)
    motion_metrics_prev = [m.item() for m in motion_metrics_prev]

    motion_metrics = [(n+p)/2 for n, p in zip(motion_metrics_next, motion_metrics_prev)]

    # epe, epe_rel, motion_scale, thres_3, thres_5, thres_10 = motion_metrics

    return motion_metrics


def flow2rgb(flow_map_np):
    '''
    flow_map_np: [H, W, 2/3]
    orginally used for optical flow visualization
    output: [H, W, 3]
    '''
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 2]
    return rgb_map.clip(0, 1)


def remap_invdepth_color(disp):
    '''
    disp: torch.Tensor [1, H, W]
    output: [H, W, 3]
    '''

    disp_np = disp.squeeze().cpu().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')

    # colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # shape [H, W, 3]
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])

    return colormapped_im


def disp2rgb(disp):
    """
    disp: torch.Tensor [N, 1, H, W]
    """
    all_rgbs = []
    for i in range(disp.shape[0]):
        t_disp = disp[i]
        t_rgb = remap_invdepth_color(t_disp) # [H, W, 3]
        all_rgbs.append(t_rgb)
    
    ret_rgb = np.stack(all_rgbs, axis=0) # [N, H, W, 3]

    return ret_rgb
    


def _gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def get_smooth_loss(preds, img):
    '''
    egde guided smoothing loss
    preds: shape [N, 1/K, H, W]
    img: shape [N, C, H, W]
    '''
    loss = 0
    B, _, H, W = img.shape
    # [N, 1, H, W]
    weights_x = torch.exp(-torch.mean(abs(_gradient_x(img)), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(abs(_gradient_y(img)), dim=1, keepdim=True))
    if isinstance(preds, list):
        for pred in preds:
            up_pred = nn.functional.interpolate(pred, size=[H, W])
            dep_dx = abs(_gradient_x(up_pred))
            dep_dy = abs(_gradient_y(up_pred))
            loss1 = torch.sum(dep_dx * weights_x) / torch.numel(dep_dx)
            loss1 += torch.sum(dep_dy * weights_y) / torch.numel(dep_dy)
            loss += loss1
    else:
        # [N, 1, H, W]
        dep_dx = abs(_gradient_x(preds))
        dep_dy = abs(_gradient_y(preds))
        loss = torch.sum(dep_dx * weights_x) / torch.numel(dep_dx)
        loss += torch.sum(dep_dy * weights_y) / torch.numel(dep_dy)
    return loss


def sparsity_loss(preds):
    """
    preds: [N, 3/1, H, W]
    """
    preds_abs = torch.abs(preds)

    preds_spatial_abs_mean = torch.mean(preds_abs, dim=[2, 3], keepdim=True).detach()

    sparse_map = 2 * preds_spatial_abs_mean * \
                torch.sqrt(preds_abs / (preds_spatial_abs_mean+1e-6) + 1)

    return torch.mean(sparse_map)


def group_smoothness(preds):
    """
    preds: [N, 1/3, H, W]
    """
    preds_dx = preds - torch.roll(preds, 1, 3)
    preds_dy = preds - torch.roll(preds, 1, 2)

    preds_dx = preds_dx[:, :, 1:, 1:]
    preds_dy = preds_dy[:,:, 1:, 1:]

    smoothness = torch.mean(torch.sqrt(1e-5 + torch.square(preds_dx) + torch.square(preds_dy)))

    return smoothness


def convert_res_sf_to_sf(K, ref_K, extrinsics, ref_extrinsics, 
                depth_map, res_sf):
    """
    depth_map: [N, 1, H, W]
    res_sf: [N, 4, H, W]:  valid_mask = res_sf[:,-1,:,:]
    """

    device = res_sf.get_device() if res_sf.get_device() >= 0 else 'cpu'
    # Project origin and reference Camera Object
    cams = Camera(K=K, Tcw=Pose(extrinsics)).to(device)
    ref_cams = Camera(K=ref_K, Tcw=Pose(ref_extrinsics)).to(device)

    scene_flow = res_sf[:, :-1, ...]
    valid_mask = res_sf[:, [-1], ...]

    # print(depth_map.type(), K.type(), res_sf.type())
    
    cam_world_points = cams.reconstruct(depth_map, frame='c')
    cam_world_points_move = cam_world_points + scene_flow.type(torch.float)
    B,_,H,W = cam_world_points_move.shape
    # Points cam frame -> world frame
    world_points = cams.Tcw @ cam_world_points_move
    cam_ref_points = ref_cams.Twc @ world_points  # shape unkown
    
    real_scene_flow = cam_ref_points - cam_world_points

    # print(world_points.shape, cam_ref_points.shape, real_scene_flow.shape)
    real_scene_flow = real_scene_flow * valid_mask

    ret = torch.cat([real_scene_flow, valid_mask], dim=1)

    return ret


def sf_sparse_loss(K, ref_K, extrinsics, ref_extrinsics, 
                depth_map, sf):
    """
    depth_map: [N, 1, H, W]
    sf: [N, 3, H, W]:  
    """
    device = sf.get_device() if sf.get_device() >= 0 else 'cpu'
    # Project origin and reference Camera Object
    cams = Camera(K=K, Tcw=Pose(extrinsics)).to(device)
    ref_cams = Camera(K=ref_K, Tcw=Pose(ref_extrinsics)).to(device)

    cam_world_points = cams.reconstruct(depth_map, frame='c')

    cam_ref_world_points = cam_world_points + sf

    cam_world_points_move = cams.Twc @ ref_cams.Tcw @ cam_ref_world_points

    res_sf = cam_world_points_move - cam_world_points

    return sparsity_loss(res_sf), res_sf


def sceneflow2rgb(sf):
    """
    from https://github.com/visinf/multi-mono-sf/blob/main/utils/output_util.py
    scene flow color coding using CIE-LAB space.
    sf: input scene flow, numpy type, size of (h, w, 3)
    output: [h,w,3]
    """

    # coordinate normalize
    max_sf = np.sqrt(np.sum(np.square(sf), axis=2)).max()
    sf = sf / max_sf

    sf_x = sf[:, :, 0]
    sf_y = sf[:, :, 1]
    sf_z = sf[:, :, 2]
    
    # rotating 45 degree
    # transform X, Y, Z -> Y, X', Z' -> L, a, b 
    sf_x_tform = sf_x * COS_45 + sf_z * SIN_45
    sf_z_tform = -sf_x * SIN_45 + sf_z * COS_45
    sf = np.stack([sf_y, sf_x_tform, sf_z_tform], axis=2) # [-1, 1] cube
    
    # norm vector to lab space: x, y, z -> z, x, y -> l, a, b
    sf[:, :, 0] = sf[:, :, 0] * 50 + 50
    sf[:, :, 1] = sf[:, :, 1] * 127
    sf[:, :, 2] = sf[:, :, 2] * 127
    
    lab_vis = lab2rgb(sf)
    lab_vis = np.uint8(lab_vis * 255)
    lab_vis = np.stack([lab_vis[:, :, 2], lab_vis[:, :, 1], lab_vis[:, :, 0]], axis=2)
    
    return lab_vis