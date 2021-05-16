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
        ratio = torch.median(pred) / torch.median(gt)
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
    