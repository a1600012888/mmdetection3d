import torch
import torch.nn as nn
import torch.nn.functional as F
from .arflow_utils import flow_warp
from .utils import get_smooth_loss

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


class UnFlowLoss(nn.Module):
    
    def __init__(self,  w_l1=0.15, w_census=0.85, reduce='min', decay=0.8):
        super(UnFlowLoss, self).__init__()
        self.photo_loss = PhotoMetricLoss(w_l1, w_census, reduce)
        self.decay = decay

    def forward(self, img_prev, img_now, img_next, flows_next, flows_prev):
        '''
        flows: list of flow of shape[ [N, 2, H, W] ]
        '''
        img_prev_back = img_prev
        img_now_back = img_now
        img_next_back = img_next

        B, _, im_h, im_w = img_now.shape
        loss_rec = 0
        loss_sm = 0
        ratio = 1.0

        output = {}
        for flow_next, flow_prev in zip(flows_next, flows_prev):
            B, _, flow_h, flow_w = flow_next.shape

            if flow_h != im_h:
                img_prev = F.interpolate(img_prev_back, (flow_h, flow_w), mode='area')
                img_now = F.interpolate(img_now_back, (flow_h, flow_w), mode='area')
                img_next = F.interpolate(img_next_back, (flow_h, flow_w), mode='area')
            else:
                img_prev, img_now, img_next = img_prev_back, img_now_back, img_next_back

            rec_from_prev = flow_warp(img_prev, flow_prev)  # backward warp
            rec_from_next = flow_warp(img_next, flow_next)

            if 'rec_from_prev' not in output:
                output['rec_from_prev'] = rec_from_prev
                output['rec_from_next'] = rec_from_next
            # we can add mask first.    
            loss = self.photo_loss(img_now, [rec_from_prev, rec_from_next])
            loss_rec = loss_rec + loss * ratio

            loss = get_smooth_loss(flow_next, img_now) + get_smooth_loss(flow_prev, img_now)
            loss_sm = loss_sm + loss * ratio
            
            ratio *= self.decay
        
        return loss_rec, loss_sm, output


            
