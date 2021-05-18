import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

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
        ratio = 1.0
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
        return [torch.zeros(1), torch.zeros(1), torch.zeros(1), ]
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
        ratio = 1.0
    num = num * 1.0
    diff_i = gt - pred

    epe_map = (diff_i) ** 2
    #print(epe_map.shape)
    epe_map = (epe_map.sum(dim=-1, ) + 1e-6) ** 0.5
    #print(epe_map.shape, 'after')
    epe = torch.mean(epe_map)
    
    epe_rel = torch.sum(epe_map / gt_speed) / num

    return [epe, epe_rel, torch.tensor(ratio)]


def flow2rgb(flow_map_np):
    '''
    flow_map_np: [H, W, 2/3]
    orginally used for optical flow visualization
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
    '''

    disp_np = disp.squeeze().cpu().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    # colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # shape [H, W, 3]
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3]) 

    return colormapped_im


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
    weights_x = torch.exp(-torch.mean(abs(_gradient_x(img)), dim=1))
    weights_y = torch.exp(-torch.mean(abs(_gradient_y(img)), dim=1))
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