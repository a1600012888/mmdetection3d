import torch
import torch.nn.functional as F
import numpy as np

from .visulize import get_painted_pts_bev

def paint(pred_mask, points, img_metas):
    '''
    Args:
        pred_mask list[torch.Tensor]: num_cams of [H, W, num_classes]
        pts list[Tensor]: list of inputs points
        sweeps list[Tensor]:
        pts_metas list[dict]
    '''
    #print('points', points[0].size(), points[0].device)
    points = points.cpu()
    num_cams = len(img_metas)
    # shape [900, 1600, 10]
    H, W, num_classes = pred_mask[0].size()
    pts3d = points[..., :3]
    num_pts, _ = points.size()
    # shape [N, 4]
    pts3d = torch.cat((pts3d, torch.ones_like(pts3d[..., :1])), -1)
    # shape [N, 4, 1]
    pts3d = pts3d.unsqueeze(dim=2)
    painted_pts = torch.cat((points, torch.zeros(num_pts, num_classes)), -1)
    for i in range(num_cams):
        # shape [4, 4]
        lidar2img = img_metas[i]['lidar2img']
        lidar2img = pts3d.new_tensor(lidar2img)
        # shape [N, 4, 4]
        lidar2img = lidar2img.repeat(num_pts, 1, 1)
        # shape [N, 4]
        pts3d_img = torch.matmul(lidar2img, pts3d).squeeze()
        pts3d_img[..., 0:2] = pts3d_img[..., 0:2] / torch.clamp(pts3d_img[..., 2:3], min=1e-5)
        pts3d_img[..., 0] /= W
        pts3d_img[..., 1] /= H
        pts3d_img[..., 0:2] = (pts3d_img[..., 0:2] - 0.5) * 2
        mask = ((pts3d_img[..., 2:3] > 1e-5)
                & (pts3d_img[..., 0:1] < 1.0)
                & (pts3d_img[..., 0:1] > -1.0)
                & (pts3d_img[..., 1:2] < 1.0)
                & (pts3d_img[..., 1:2] > -1.0))
        #print('num_pts', pts3d_img[..., 0:1][mask].numel())
        # shape [1, num_classes, H, W]
        label_map = pred_mask[i].permute(2, 0, 1).cpu().unsqueeze(dim=0)
        #print('label_map', label_map.size(), 'pred_mask', pred_mask[i].size())
        # shape [1, N, 1, 2]
        pts3d_img = pts3d_img.unsqueeze(dim=1).unsqueeze(dim=0).cpu()[..., :2]
        # shape [1, num_classes, N, 1]
        sampled_mask = F.grid_sample(label_map, pts3d_img, mode='nearest')
        sampled_mask = sampled_mask.squeeze().permute(1, 0)
        ind = (sampled_mask == 1) & mask
        #print('num_painted_pts', ind.sum())
        painted_pts[:, 5:][ind] = 1
    # random pick one label if a point is projected to different images and painted one more times
    mul_labels = painted_pts[:, 5:].sum(dim=1)
    mul_labels_index = (mul_labels > 1).nonzero()
    #print('painted_pts', painted_pts.size())
    for index in mul_labels_index:
        index = index.item()
        fir_cls = 0
        for cls in range(num_classes):
            if painted_pts[index][5+cls] > 0.5:
                fir_cls = cls
                break
        painted_pts[index][6+fir_cls:] = 0.0
    return painted_pts


def paint_pts(pred_mask, points, sweeps, img_metas, pts_metas):
    #print('points', points.size())
    data_root = 'data/nuscenes/paintedpoints'
    painted_pts = paint(pred_mask, points[0], img_metas)
    #get_painted_pts_bev(painted_pts)
    filename_split = img_metas[0]['pts_filename'].split('/')
    filename = data_root + '/' + filename_split[4] + '/' + filename_split[5] + '/' + filename_split[6]
    #print('points', img_metas[0]['pts_filename'])
    #print('points', filename)
    np.save(filename, painted_pts)
    reloaded_pts = np.load(filename + '.npy')
    print(painted_pts.size(), reloaded_pts.shape)
    if len(sweeps) != len(pts_metas):
        print(len(sweeps), len(pts_metas))
        raise ValueError('Expected {} but got {}',format(len(sweeps), len(pts_metas)))
    for i in range(len(sweeps)):
        sweep = sweeps[i].squeeze()
        painted_pts = paint(pred_mask, sweep, img_metas)
        #print(pts_metas[i].keys())
        # transform sweep points into origin coordination
        painted_pts[:, :3] -= pts_metas[i]['sensor2lidar_translation']
        painted_pts[:, :3] = painted_pts[:, :3] @ np.linalg.pinv(pts_metas[i][
                        'sensor2lidar_rotation'].T)
        filename_split = pts_metas[i]['data_path'].split('/')
        filename = data_root + '/' + filename_split[4] + '/' + filename_split[5] + '/' + filename_split[6]
        #print('sweeps', pts_metas[i]['data_path'])
        #print('sweeps', filename)
        #get_painted_pts_bev(painted_pts)
        np.save(filename, painted_pts)