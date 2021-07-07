
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    
    x_ = x.clamp(min=0, max=1)
    x1 = x_.clamp(min=eps)
    x2 = (1 - x_).clamp(min=eps)
    ret = torch.log(x1 / x2)
    
    return ret


@ATTENTION.register_module()
class Detr3DCamCrossAttenOffsets(BaseModule):
    """An attention module used in Deformable-Detr. `Deformable DETR:
    Deformable Transformers for End-to-End Object Detection.
      <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 use_dconv=False,
                 use_level_cam_embed=False,
                 norm_cfg=None,
                 init_cfg=None):
        super(Detr3DCamCrossAttenOffsets, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.dim_per_head = dim_per_head
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed

        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_heads*num_points*num_levels)
        self.offsets = nn.Linear(embed_dims, num_heads*num_points*3)

        self.value_proj = nn.Linear(embed_dims,
                                    embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.embed_dims, self.num_levels))
        self.cam_embeds =  nn.Parameter(
            torch.Tensor(self.embed_dims, self.num_cams))

        # dynamic conv
        self.dynamic_dims = 64
        self.dynamic_layer = nn.Linear(self.embed_dims, self.dynamic_dims * self.embed_dims * 2)
        self.norm1 = nn.LayerNorm(self.dynamic_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)
        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(self.embed_dims*num_cams*num_levels*num_points, self.embed_dims)
        self.norm3 = nn.LayerNorm(self.embed_dims)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        #constant_init(self.offsets, 0.)
        xavier_init(self.offsets, distribution='uniform', bias=0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_cams)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1,
                         2).repeat(1, self.num_points, 1)
        grid_init = torch.cat((grid_init, grid_init.new_zeros((*grid_init.shape[:-1], 1))), dim=-1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1

        # self.offsets.bias.data = grid_init.view(-1) / self.num_points *0.01
        constant_init(self.attention_weights, val=0., bias=0.)
        #constant_init(self.offsets, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        normal_(self.cam_embeds)

    def dynamic_conv(self, query, feats):
        """query: (b, n, c).
           feats: (b, c, n, num_cam, num_point, num_level).
        """
        B, num_query = query.size()[0:2]
        parameters = self.dynamic_layer(query) # (b, n, c, c*dynamic_dims)
        param1 = parameters[:, :, :self.embed_dims*self.dynamic_dims].view(B, num_query, self.embed_dims, self.dynamic_dims)
        param2 = parameters[:, :, self.embed_dims*self.dynamic_dims:].view(B, num_query, self.dynamic_dims, self.embed_dims)
        feats = feats.permute(0, 2, 3, 4, 5, 1).reshape(
            B, num_query, self.num_cams*self.num_points*self.num_levels, self.embed_dims)
        feats = feats @ param1
        feats = self.norm1(feats)
        feats = self.activation(feats)
        feats = feats @ param2
        feats = self.norm2(feats)
        feats = self.activation(feats)
        feats = feats.reshape(B, num_query, self.num_cams*self.num_points*self.num_levels*self.embed_dims)
        feats = self.out_layer(feats)
        feats = self.norm3(feats)
        feats = self.activation(feats)
        return feats

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value  Tensor with shape `[[B, num_cam, C, H, W]]`
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        #print('value shape!!', value[0].shape, len(value))
        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # value projection
        value_proj_list = []
        for lvl, feat in enumerate(value):
            _B, _N, _C, _H, _W = feat.shape
            # feat: [B, N, C, H, W] => [B, N, H, W, C]
            value_flat = feat.permute(0, 1, 3, 4, 2)
            value_flat_proj = self.value_proj(value_flat)

            # [B, N, H, W, num_heads, dim_per_head]
            value_flat_proj = value_flat_proj.view(_B, _N, _H, _W, self.num_heads, -1)

            value_proj_list.append(value_flat_proj)
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        # [bs, num_query, num_cams*num_heads*num_points*num_levels]
        attention_weights = self.attention_weights(query)

        # [B, num_query, num_heads*num_points*3]
        # shape [B, 1, num_query, 1, num_heads, num_points, 3]
        offsets = self.offsets(query).view(bs, 1, num_query, 1, self.num_heads, self.num_points, 3) * 0.01  # or not?
        # shape [B, num_cam, num_query, num_level, num_heads, num_points, 3]
        offsets = offsets.repeat(1, self.num_cams, 1, self.num_levels, 1, 1, 1)

        # [B, num_query, num_heads, dim_per_head/1, num_cam, num_points, num_level]
        # print(offsets.mean().item(), reference_points.mean().item(), inverse_sigmoid(reference_points).mean().item())
        output, mask = feature_sampling(value_proj_list, reference_points, offsets, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        # [embed_dims, num_level]
        level_embeds = self.level_embeds.view(1, 1, self.num_heads, self.dim_per_head, 1, 1,  self.num_levels)
        # [embed_dims, num_cams]
        cam_embeds = self.cam_embeds.view(1, 1, self.num_heads, self.dim_per_head, self.num_cams, 1, 1)

        if self.use_level_cam_embed:
            output = output + level_embeds + cam_embeds

        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams * self.num_levels)
        # attention_weights = attention_weights.softmax(-1)
        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams, self.num_levels)

        # TODO: use if else to switch between dynamic conv and weighted sum

        if self.use_dconv:
            print('using dconv!!')
            output = output * mask
            output = self.dynamic_conv(query, output)
            output = output.permute(1, 0, 2)
        else:
            # [bs, num_query, num_cams*num_heads*num_points*num_levels]
            # attention_weights = (attention_weights / torch.sqrt(output.new_tensor(self.embed_dims*1.0))).softmax(-1)
            attention_weights = attention_weights.softmax(-1)

            # [bs, num_query, self.num_heads, 1, self.num_cams, self.num_points, self.num_levels]
            attention_weights = attention_weights.view(
                bs, num_query, self.num_heads, 1, self.num_cams, self.num_points, self.num_levels)

            # [bs, num_query, num_heads, 1, num_cams, num_points, num_levels]
            # attention_weights = self.weight_dropout(attention_weights.sigmoid()) * mask.detach()
            attention_weights = attention_weights * mask.detach()
            # [B, num_query, num_heads, dim_per_head, num_cam, num_points, num_level]
            output = output * attention_weights
            
            # [B, num_query, num_heads, dim_per_head]
            output = output.sum(-1).sum(-1).sum(-1) / (attention_weights.sum(-1).sum(-1).sum(-1).sum(-1, keepdim=True) + 1e-7)
            # output = output.sum(-1).sum(-1).sum(-1) / (mask.sum(-1).sum(-1).sum(-1).sum(-1, keepdim=True) + 1e-4)

            # [B, num_query, C]
            output = torch.flatten(output, 2,3)
            # [bs, num_query, C] => [num_query, bs, C]
            output = output.permute(1, 0, 2)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)

        return self.dropout(output) + inp_residual


def feature_sampling(mlvl_feats, reference_points, offsets, pc_range, img_metas):
    '''
    mlvl_feats: [[B, num_cam, H, W, num_heads, dim_per_head]]
    reference_points: [B, num_query, 4]
    offsets: [B, num_cam, num_query, num_level, num_heads, num_points, 3]
    '''
    lidar2img = []
    img_flip = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
        if 'img_flip' in img_metas:
            img_flip.append(img_meta['img_flip'])
        else:
            img_flip.append(np.zeros((6,)))
    lidar2img = np.asarray(lidar2img)
    img_flip = np.asarray(img_flip)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    img_flip = reference_points.new_tensor(img_flip) # (B, N)
    #reference_points = reference_points.clone() # (B, num_query, 3)

    reference_points = inverse_sigmoid(reference_points)
    B, num_query, _ = reference_points.shape
    reference_points = reference_points.view(B, 1, num_query, 1, 1, 1, _)
    B, num_cam, _, num_level, num_heads, num_points, _ = offsets.shape

    # [B, num_cam, num_query, num_level, num_heads, num_points, 3]
    reference_points = reference_points.repeat(1, num_cam, 1, num_level, num_heads, num_points, 1)
    reference_points = reference_points + offsets
    reference_points = reference_points.sigmoid() # inplace op below; 
    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_cam, num_query, num_level, num_heads, num_points, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_cam, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)
    # [B, num_cam, num_query, num_level, num_heads, num_points, 4, 1]
    reference_points = reference_points.unsqueeze(-1)
    # [B, num_cam, 4, 4] => [B, num_cam, num_query, num_level, num_heads, num_points, 4, 4]
    lidar2img = lidar2img.view(B, num_cam, 1, 1, 1, 1, 4, 4).repeat(1, 1, num_query, num_level, num_heads, num_points, 1, 1)
    # [B, num_cam, num_query, num_level, num_heads, num_points, 4]
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    mask = (reference_points_cam[..., 2:3] > 0)
    reference_points_cam = reference_points_cam[..., 0:2] / (reference_points_cam[..., 2:3] + 1e-5)
    reference_points_cam[..., 0] /= img_metas[0]['pad_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['pad_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    # [B, num_cam, num_query, num_level, num_heads, num_points, 1]
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))

    #print('ref, mask!!! lidar2img', reference_points_cam.shape, mask.shape, lidar2img.shape)
    #mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    sampled_feats = []

    img_flip = img_flip.view(B, num_cam, 1, 1, 1)

    for lvl, feat in enumerate(mlvl_feats):
        B, _num_cam, H, W, _num_heads, dim_per_head, = feat.size()
        # [B, num_cam, num_heads, dim_per_head, H, W]
        feat = feat.permute(0, 1, 4, 5, 2, 3)
        # [B * num_cam * num_heads, dim_per_head, H, W]
        feat = torch.flatten(feat, start_dim=0, end_dim=2)

        ## feat_flip = torch.flip(feat, [-1])
        ## feat = torch.where(img_flip > 0.0, feat_flip, feat)


        # [B, num_cam, num_query, num_heads, num_points, 2]
        reference_points_cam_lvl = reference_points_cam[:, :, :, lvl, ...]
        # [B, num_cam, num_heads, num_query, num_points, 2]
        reference_points_cam_lvl = reference_points_cam_lvl.permute(0, 1, 3, 2, 4, 5)
        # [B * num_cam * num_heads, num_query, num_points, 2]
        reference_points_cam_lvl = torch.flatten(reference_points_cam_lvl, 0, 2)

        # [B * num_cam * num_heads, dim_per_head, num_query, num_points]
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, 
                                    mode='bilinear',
                                    padding_mode='zeros',
                                    align_corners=False)

        # [B, num_cam, num_heads, dim_per_head, num_query, num_points]
        sampled_feat = sampled_feat.unflatten(0, (B, num_cam, num_heads))

        # [B, num_query, num_heads, dim_per_head, num_cam, num_points]
        sampled_feat = sampled_feat.permute(0, 4, 2, 3, 1, 5)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)

    # sample_feats: [B, num_query, num_heads, dim_per_head, num_cam, num_points, num_level]
    # mask: [B, num_cam, num_query, num_level, num_heads, num_points, 1]
    # now mask: [B, num_query, num_heads, 1, num_cam, num_points, num_level]
    mask = mask.permute(0, 2, 4, 6, 1, 5, 3)

    return sampled_feats, mask
