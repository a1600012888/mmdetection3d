import torch
import torch.nn.functional as funct
from ..geometry.camera import Camera
from ..geometry.pose import Pose
from ..geometry.camera_utils import view_synthesis
from ..utils.depth import inv2depth, depth2inv


padding_mode = 'zeros' # padding mode for view_synthesis
# Camera order swap
left_swap = [i for i in range(-1,5)]
right_swap = [i % 6 for i in range(1,7)]

def warp_ref_image_temporal(inv_depth, ref_image, K, ref_K, extrinsics, ref_extrinsics,
                            scene_flow):
    """
    Warps a reference image to produce a reconstruction of the original one (temporal-wise).

    Parameters
    ----------
    inv_depths : torch.Tensor [B,1,H,W]
        Inverse depth map of the original image
    ref_image : torch.Tensor [B,3,H,W]
        Reference RGB image
    K : torch.Tensor [B,3,3]
        Original camera intrinsics
    ref_K : torch.Tensor [B,3,3]
        Reference camera intrinsics
    extrinsics: torch.Tensor [B,4,4] C->W
        Extrinsic of transform the point from Camera frame to World frame (at time t)
    ref_extrinsics: torch.Tensor [B,4,4] C->W
        Extrinsic of transform the point from Camera frame to World frame (at time t-1 or t+1)
    scene_flow : torch.Tensoe [B,3,H,W]
        Scene_flow in Origin Camera frame
        t->t-1 or t->t+1

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image (reconstructing the original one)
    """
    B, _, H, W = ref_image.shape
    device = ref_image.get_device() if ref_image.get_device() >= 0 else 'cpu'
    # Project origin and reference Camera Object
    cams = Camera(K=K, Tcw=Pose(extrinsics)).to(device)
    ref_cams = Camera(K=ref_K, Tcw=Pose(ref_extrinsics)).to(device)
    # View synthesis
    depth = inv2depth(inv_depth)
    ref_warped,ref_coords = view_synthesis(ref_image, depth, ref_cams, cams, scene_flow,
                              padding_mode=padding_mode)
    return ref_warped


def warp_ref_image_spatial(inv_depth, ref_image, K, ref_K, extrinsics_1, extrinsics_2):
    """
    Warps a reference image to produce a reconstruction of the original one (spatial-wise).

    Parameters
    ----------
    inv_depths : torch.Tensor [6,1,H,W]
        Inverse depth map of the original image
    ref_image : torch.Tensor [6,3,H,W]
        Reference RGB image
    K : torch.Tensor [B,3,3]
        Original camera intrinsics
    ref_K : torch.Tensor [B,3,3]
        Reference camera intrinsics
    extrinsics_1: torch.Tensor [B,4,4] c->w
        target image extrinsics
    extrinsics_2: torch.Tensor [B,4,4] c->w
        context image extrinsics
    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image (reconstructing the original one)
    valid_points_mask : of torch.Tensor [B,1,H,W]
        valid points mask
    """
    B, _, H, W = ref_image.shape
    device = ref_image.get_device() if ref_image.get_device() >= 0 else 'cpu'
    # Generate cameras for all scales
    cam = Camera(K=K, Tcw=Pose(extrinsics_1)).to(device)
    ref_cam = Camera(K=ref_K, Tcw=Pose(extrinsics_2)).to(device)
    # View synthesis
    depth = inv2depth(inv_depth)
    ref_warped, ref_coord = view_synthesis(ref_image, depth, ref_cam, cam,
                          scene_flow=None, padding_mode=padding_mode)
    # Calculate valid_points_mask
    valid_points_mask = ref_coord.abs().max(dim=-1)[0] <= 1
    return ref_warped, valid_points_mask


def warp_ref_image_temporal_spatial(inv_depth, ref_image, K, ref_K, extrinsics_1, extrinsics_2,scene_flow):
    """
    Warps a reference image to produce a reconstruction of the original one (spatial-wise).

    Parameters
    ----------
    inv_depths : torch.Tensor [6,1,H,W]
        Inverse depth map of the original image
    ref_image : torch.Tensor [6,3,H,W]
        Reference RGB image
    K : torch.Tensor [B,3,3]
        Original camera intrinsics
    ref_K : torch.Tensor [B,3,3]
        Reference camera intrinsics
    extrinsics_1: torch.Tensor [B,4,4] c->w
        target image extrinsics
    extrinsics_2: torch.Tensor [B,4,4] c->w
        context image extrinsics
    scene_flow: torch.Tensor [B,3,H,W]
        scen_flow of target image at time t
    delta_t: float
        time of one frame
    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image (reconstructing the original one)
    valid_points_mask : of torch.Tensor [B,1,H,W]
        valid points mask
    """
    B, _, H, W = ref_image.shape
    device = ref_image.get_device() if ref_image.get_device() >= 0 else 'cpu'
    # Generate cameras for all scales
    cam = Camera(K=K, Tcw=Pose(extrinsics_1)).to(device)
    ref_cam = Camera(K=ref_K, Tcw=Pose(extrinsics_2)).to(device)
    # View synthesis
    depth = inv2depth(inv_depth)
    ref_warped, ref_coord = view_synthesis(ref_image, depth, ref_cam, cam,
                          scene_flow=scene_flow, padding_mode=padding_mode)
    # Calculate valid_points_mask
    valid_points_mask = ref_coord.abs().max(dim=-1)[0] <= 1
    return ref_warped, valid_points_mask


'''
def calc_scene_flow_consistency_loss(scene_flows, intrinsic, extrinsic, inv_depth):
    """
    Calculates the consistency loss for multi-cameras pose
    Parameters
    ----------
    scene_flows : list of torch.Tensor [[6,3,H,W], [6,3,H,W]]
        Predicted scene flows for 6 cameras from the sceneflowNet [[scene_flows(t->t-1)], [scene_flows(t->t+1)]]
    extrinsics: torch.Tensor [6,4,4] c->w
        Extrinsics matrix for 6 cameras
    inv_depths: torch.Tensor [6,1,H,W]

    Returns
    -------
    consistency_loss : torch.Tensor [1]
        Consistency loss
    """

    assert inv_depth.dim() > 2, "Depth should have 4 dim, (B,1,H,W)"
    device = inv_depth.get_device() if inv_depth.get_device()>=0 else 'cpu'

    # Construct ref intrinsics, extrinsics in order: [right,left]
    extrinsics_1 = extrinsic
    extrinsics_2 = [extrinsic[right_swap,...], extrinsic[left_swap,...]]
    intrinsics_1 = intrinsic
    intrinsics_2 = [intrinsic[right_swap,...], intrinsic[left_swap,...]]
    ref_scene_flows = [] # [t->t-1,t->t+1], [[right,left],[right,left]]
    for i in range(len(scene_flows)):
        ref_scene_flows.append([scene_flows[i][right_swap,...],scene_flows[i][left_swap,...]])

    # inv_depths -> depths
    depth = inv2depth(inv_depth)
    B, _, DH, DW = depth.shape
    consistency_losses = []
    cam = Camera(K=intrinsics_1, Tcw=Pose(extrinsics_1)).to(device) # origin cam
    # Calculate right camera scene flow consistency loss
    for i in range(2):  # iterate over t-1 and t+1
        ref_cam = Camera(K=intrinsics_2[i], Tcw=Pose(extrinsics_2[i])).to(device)
        scene_flow = scene_flows[i]  # [6,3,H,W] t-1 or t+1
        for j, ref_scene_flow in enumerate(ref_scene_flows[i]):  # iterate over left or right
            # Project target image to ref image
            cam_points = cam.reconstruct(depth, frame='c')
            world_points_from_cam = cam.Tcw @ cam_points
            ref_coords = ref_cam.project(world_points_from_cam, frame='w')
            # Interpolate scene flow
            warped_target_scene_flow_ref = funct.grid_sample(ref_scene_flow, ref_coords, mode='bilinear',
                                 padding_mode=padding_mode, align_corners=True)
            # Project scene flow to ref cam frame
            target_scene_flow_ref = (extrinsics_2[j][:, :3, :3].permute(0,2,1)
                                @ extrinsics_1[:, :3, :3]
                                @ scene_flow.view(B,-1,DH*DW)).view(B,-1,DH,DW)
            valid_points_mask = ref_coords.abs().max(dim=-1)[0] <= 1
            consistency_losses.append(torch.abs(warped_target_scene_flow_ref-target_scene_flow_ref).mean(1).unsqueeze(1)
                                         *valid_points_mask)
    return reduce_loss(torch.cat(consistency_losses,1))
'''


def calc_scene_flow_consistency_loss(scene_flows, intrinsic, extrinsic, inv_depth):
    """
    Calculates the consistency loss for multi-cameras pose
    Parameters
    ----------
    scene_flows : list of torch.Tensor [[6,3,H,W], [6,3,H,W]]
        Predicted scene flows for 6 cameras from the sceneflowNet [[scene_flows(t->t-1)], [scene_flows(t->t+1)]]
    extrinsics: torch.Tensor [6,4,4] c->w
        Extrinsics matrix for 6 cameras
    inv_depths: torch.Tensor [6,1,H,W]

    Returns
    -------
    consistency_loss : torch.Tensor [1]
        Consistency loss
    """

    assert inv_depth.dim() > 2, "Depth should have 4 dim, (B,1,H,W)"
    device = inv_depth.get_device() if inv_depth.get_device()>=0 else 'cpu'

    # Construct ref intrinsics, extrinsics in order: [right,left]
    extrinsics_1 = extrinsic
    extrinsics_2 = [extrinsic[right_swap,...], extrinsic[left_swap,...]]
    intrinsics_1 = intrinsic
    intrinsics_2 = [intrinsic[right_swap,...], intrinsic[left_swap,...]]
    ref_scene_flows = [] # [t->t-1,t->t+1], [[right,left],[right,left]]
    for i in range(len(scene_flows)):
        ref_scene_flows.append([scene_flows[i][right_swap,...],scene_flows[i][left_swap,...]])

    # inv_depths -> depths
    depth = inv2depth(inv_depth)
    B, _, DH, DW = depth.shape
    consistency_losses = []
    cam = Camera(K=intrinsics_1, Tcw=Pose(extrinsics_1)).to(device) # origin cam
    # Calculate right camera scene flow consistency loss
    valid_points_ratios = []
    for time in range(2):  # iterate over t->t-1 and t->t+1
        scene_flow = scene_flows[time]  # scene flow of origin cam t->t-1 or t->t+1
        for l_r, ref_scene_flow in enumerate(ref_scene_flows[time]):  # iterate over left and right
            # Project target image to ref image
            ref_cam = Camera(K=intrinsics_2[l_r].float(), Tcw=Pose(extrinsics_2[l_r])).to(device) # left or right cam
            cam_points = cam.reconstruct(depth, frame='c')
            world_points_from_cam = cam.Tcw @ cam_points
            ref_coords = ref_cam.project(world_points_from_cam, frame='w')
            # Interpolate scene flow
            warped_target_scene_flow_ref = funct.grid_sample(ref_scene_flow, ref_coords, mode='bilinear',
                                    padding_mode=padding_mode, align_corners=True)
            # Project scene flow to ref cam frame
            target_scene_flow_ref = (extrinsics_2[l_r][:, :3, :3].permute(0,2,1)
                                @ extrinsics_1[:, :3, :3]
                                @ scene_flow.view(B,-1,DH*DW)).view(B,-1,DH,DW)
            valid_points_mask = ref_coords.abs().max(dim=-1)[0] <= 1
            valid_points_ratios.append(valid_points_mask.type(torch.float).mean().item())
            consistency_losses.append(torch.abs(warped_target_scene_flow_ref-target_scene_flow_ref).mean(1).unsqueeze(1)
                                            *valid_points_mask)
    valid_points_ratio = sum(valid_points_ratios) / len(valid_points_ratios)

    return reduce_loss(torch.cat(consistency_losses,1)), valid_points_ratio

def calc_depth_consistency_loss(inv_depth, intrinsic, extrinsic):
    """
    Calculates the consistency loss for multi-cameras pose
    Parameters
    ----------
    inv_depth :[6,1,H,W]
    intrinsic: torch.Tensor [6,3,3] 
        intrinsic matrix for 6 cameras
    extrinsics: torch.Tensor [6,4,4] c->w
        Extrinsics matrix for 6 cameras

    Returns
    -------
    consistency_loss : torch.Tensor [1]
    valid_points_ratio: float, measure the percent of valid pixels
        Consistency loss
    """

    assert inv_depth.dim() > 2, "Depth should have 4 dim, (B,1,H,W)"
    device = inv_depth.get_device() if inv_depth.get_device()>=0 else 'cpu'

    # Construct ref intrinsics, extrinsics in order: [right,left]
    extrinsics_1 = extrinsic
    extrinsics_2 = [extrinsic[right_swap,...], extrinsic[left_swap,...]]
    intrinsics_1 = intrinsic
    intrinsics_2 = [intrinsic[right_swap,...], intrinsic[left_swap,...]]

    ref_inv_depths = [
        inv_depth[right_swap, ...], 
        inv_depth[left_swap, ...], 
    ]
    # inv_depths -> depths
    depth = inv2depth(inv_depth)
    B, _, DH, DW = depth.shape
    consistency_losses = []
    cam = Camera(K=intrinsics_1, Tcw=Pose(extrinsics_1)).to(device) # origin cam
    # Calculate right camera scene flow consistency loss
    
    valid_points_ratios = []
    for i in range(2):
        ref_cam = Camera(K=intrinsics_2[i], Tcw=Pose(extrinsics_2[i])).to(device)
        ref_inv_depth = ref_inv_depths[i]
    
        # Project target image to ref image to get ref_cords
        world_points_from_cam = cam.reconstruct(depth, frame='w')
        ref_coords = ref_cam.project(world_points_from_cam, frame='w')
        
        # transform ref_depth to tgt view
        world_points_from_ref_cam = ref_cam.reconstruct(inv2depth(ref_inv_depth), frame='w')
        points_to_tgt_camera = cam.Twc @  world_points_from_ref_cam
        ref_invdepth_in_tft_camera = depth2inv(points_to_tgt_camera[:, [-1], ...])

        # warp ref inv depth to reconstruct tgt depth
        warped_target_inv_depth = funct.grid_sample(ref_invdepth_in_tft_camera, ref_coords, mode='bilinear',
                                padding_mode=padding_mode, align_corners=True)

        valid_points_mask = ref_coords.abs().max(dim=-1)[0] <= 1
        # record the ratio of valid pixels
        valid_points_ratios.append(valid_points_mask.type(torch.float).mean().item())
        consistency_losses.append(torch.abs(warped_target_inv_depth - inv_depth).mean(1).unsqueeze(1)
                                        *valid_points_mask)

    valid_points_ratio = sum(valid_points_ratios) / len(valid_points_ratios)

    return reduce_loss(torch.cat(consistency_losses,1)), valid_points_ratio


def reduce_loss(losses_reduce, op='min'):
    """
    Combine loss

    Parameters
    ----------
    losses_reduce : list of torch.Tensor [B,?,H,W]

    op: 'mean' or 'min'

    Returns
    -------
    loss_reduced : torch.Tensor [1]
        Reduced loss
    """

    # Reduce function
    def reduce_function(losses):
        if op == 'mean':
            return losses.mean()
        elif op == 'min':
            return losses.min(1, True)[0].mean()
        else:
            raise NotImplementedError(
                'Unknown photometric_reduce_op: {}'.format(op))

    # Reduce loss
    loss_reduced = sum([reduce_function(losses_reduce[i])
                            for i in range(len(losses_reduce))]) / len(losses_reduce)
    return loss_reduced
