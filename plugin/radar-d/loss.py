import torch
import torch.nn as nn
import torch.nn.functional as F
from .warp_utils import opflow_from_sf, get_occu_mask_bidirection, flow_warp


class PhotoMetricLoss(nn.Module):

    def __init__(self, w_l1=0.5, w_census=0.5):
        super().__init__()
        self.w_l1 = w_l1
        self.w_census = w_census
        
    def forward(self, img, rec_img, mask):
        loss = 0
        if self.w_l1 > 0:
            
            l1_loss_map = torch.mean(torch.abs(img - rec_img), dim=1) * mask
            l1_loss = torch.mean(l1_loss_map)
            
            loss = loss + self.w_l1 * l1_loss
        if self.w_census > 0:
            
            census_loss_map = TernaryLoss(img, rec_img) * mask
            census_loss = torch.mean(census_loss_map)

            loss = loss + self.w_census * census_loss
    
        return loss            
            

def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


class BidirectionalSFRecLoss(nn.Module):

    def __init__(self, w_l1=0.5, w_census=0.5,  decay=0.8):
        super().__init__()
        self.photo_loss = PhotoMetricLoss(w_l1, w_census)
        self.decay = decay
    
    def forward(self, sf, rgbd1, rgbd2, cam_intrin1, cam_intrin2):
        visual_map = {}
        sf12, sf21 = sf[:, :3], sf[:, 3:]
        d1, d2 = rgbd1[:, [3]], rgbd2[:, [3]]
        rgb1, rgb2 = rgbd1[:, :3], rgbd2[:, :3]

        flow12 = opflow_from_sf(d1, sf12, cam_intrin1, cam_intrin2)
        flow21 = opflow_from_sf(d2, sf21, cam_intrin2, cam_intrin1)

        # >0 means not occuluded
        mask1 = 1 - get_occu_mask_bidirection(flow12, flow21, scale=0.2, bias=2.0)
        mask2 = 1 - get_occu_mask_bidirection(flow21, flow12, scale=0.2, bias=2.0)

        rec_rgbd1 = flow_warp(rgbd2, flow12, pad='zeros', mode='bilinear')
        rec_rgbd2 = flow_warp(rgbd1, flow21, pad='zeros', mode='bilinear')

        rec_rgb1, rec_rgb2 = rec_rgbd1[:, :3], rec_rgbd2[:, :3]
        rec_d1, rec_d2 = rec_rgbd1[:, [3]], rec_rgbd2[:, [3]]
        delta_z1, delta_z2 = rec_d1 - d1, rec_d2 - d2

        photo_loss = self.photo_loss(rgb1, rec_rgb1, mask1) + \
                        self.photo_loss(rgb2, rec_rgb2, mask2) 
         
        z_loss = torch.mean((delta_z1 - sf12[:,[2]]).abs()) + \
                    torch.mean((delta_z2 - sf21[:,[2]]).abs())
        
        valid_ratio = torch.mean(torch.cat([mask1, mask2], dim=1))
        visual_map['now2next_img'] = rec_rgb2
        visual_map['next2now_img'] = rec_rgb1
        visual_map['now2next_flow'] = flow12
        visual_map['next2now_flow'] = flow21
        visual_map['mask1'] = mask1
        visual_map['mask2'] = mask2
        return photo_loss, z_loss, valid_ratio, visual_map
        

