import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotoMetricLoss(nn.Module):

    def __init__(self, w_l1=0.15, w_census=0.85, reduce='min'):
        super().__init__()
        self.w_l1 = w_l1
        self.w_census = w_census
        self.reduce = reduce
    
    def forward(self, img, pred_list):
        loss = 0
        if self.w_l1 > 0:
            l1_loss_maps = [torch.mean(torch.abs(img_pred - img), dim=1, keepdim=True) for img_pred in pred_list]
            
            l1_loss_map = torch.cat(l1_loss_maps, dim=1)
            if self.reduce == 'min':
                l1_loss = torch.mean(torch.min(l1_loss_map, dim=1)[0])
            else:
                l1_loss = torch.mean(l1_loss_map)
            
            loss = loss + self.w_l1 * l1_loss
        if self.w_census > 0:
            
            census_loss_maps = [TernaryLoss(img, img_pred) for img_pred in pred_list]
            
            census_loss_map = torch.cat(census_loss_maps, dim=1)

            if self.reduce == 'min':
                census_loss = torch.mean(torch.min(census_loss_map, dim=1)[0])
            else:
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