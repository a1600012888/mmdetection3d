import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

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
    