# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as funct

########################################################################################################################


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""
    return torch.tensor([[fx,  0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

########################################################################################################################


def view_synthesis(ref_image, depth, ref_cam, cam, scene_flow,
                   mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    scene_flow: torch.Tensor [B,3,H,W]
        scene_flow in shape of [B,3,H,W]
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstrcut points in cam frame
    cam_world_points = cam.reconstruct(depth, frame='c')
    if scene_flow is not None:
        cam_world_points += scene_flow
        ref_coords = ref_cam.project(cam_world_points, frame='c')
    else:
        B,_,H,W = cam_world_points.shape
        # Points cam frame -> world frame
        world_points = cam.Tcw @ cam_world_points
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True), ref_coords.unsqueeze(1)

########################################################################################################################