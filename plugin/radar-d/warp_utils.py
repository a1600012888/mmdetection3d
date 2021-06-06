import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_intrin(K):
    '''
    Inverse intrinsics (for lifting)
    K: [N, 3, 3]
    '''
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    Kinv = K.clone()
    Kinv[:, 0, 0] = 1. / fx
    Kinv[:, 1, 1] = 1. / fy
    Kinv[:, 0, 2] = -1. * cx / fx
    Kinv[:, 1, 2] = -1. * cy / fy
    return Kinv


def mesh_grid(B, H, W, normalized=False):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    if normalized:
        x_base = 2*x_base / W - 1.0
        y_base = 2*y_base / H - 1.0
    
    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='zeros', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=False)
    
    return im1_recons


def opflow_from_sf(depth, sf, cam_intrin1, cam_intrin2):
    B, _, H, W = depth.shape
    grid_2d = mesh_grid(B, H, W, normalized=False).type_as(depth)
    ones = torch.ones_like(depth)

    grid_3d = torch.cat([grid_2d, ones], dim=1)

    grid_3d_flat = grid_3d.view(B, 3, -1)

    xnorm = (inverse_intrin(cam_intrin1).bmm(grid_3d_flat)).view(B, 3, H, W)
        
    Xc = xnorm * depth

    Xc_2 = Xc + sf
    ref_cords = cam_intrin2.bmm(Xc_2.view(B, 3, -1))

    Z = ref_cords[:, 2].clamp(min=1e-5)
    Xcord = (ref_cords[:, 0] / Z).view(B, 1, H, W)
    Ycord = (ref_cords[:, 1] / Z).view(B, 1, H, W)

    ref_cord_2d = torch.cat([Xcord, Ycord], dim=1)

    opflow = ref_cord_2d - grid_2d

    return opflow


def get_occu_mask_bidirection(flow12, flow21, scale=0.2, bias=2.0):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float().detach()



