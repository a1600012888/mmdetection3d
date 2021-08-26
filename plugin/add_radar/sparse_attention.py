
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_
from mmcv import ConfigDict, deprecated_api_warning
import warnings
import math

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
class Detr3DCamRadarSparseCrossAtten(BaseModule):
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
                 radar_dims=64, 
                 radar_topk=3, 
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 drop_img=0.0, 
                 drop_radar=0.0, 
                 use_level_cam_embed=False,
                 norm_cfg=None,
                 init_cfg=None, 
                 batch_first=False, 
                 **kwargs, 
                 ):
        super(Detr3DCamRadarSparseCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.drop_img = nn.Dropout(drop_img)
        self.drop_radar = nn.Dropout(drop_radar)
        self.pc_range = pc_range
        self.radar_dims = radar_dims

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

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_level_cam_embed = use_level_cam_embed
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        self.attention_weights_radar = nn.Linear(embed_dims, radar_topk)
        self.radar_topk = radar_topk

        self.radar_out_proj = nn.Sequential(
            nn.Linear(self.radar_dims, self.radar_dims), 
            nn.LayerNorm(self.radar_dims),
            nn.ReLU(inplace=True), 
        )

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims+radar_dims, embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )
        
        if self.use_level_cam_embed:
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.embed_dims, self.num_levels))
            self.cam_embeds = nn.Parameter(
                torch.Tensor(self.embed_dims, self.num_cams))

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_level_cam_embed:
            normal_(self.level_embeds)
            normal_(self.cam_embeds)
    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                radar_feats=None,  # [B, M, K]
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
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

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        if self.use_level_cam_embed:
            level_embeds = self.level_embeds.view(1, self.embed_dims, 1, 1, 1, self.num_levels)
            cam_embeds = self.cam_embeds.view(1, self.embed_dims, 1, self.num_cams, 1, 1)
            output = output + level_embeds + cam_embeds

        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams * self.num_levels)
        # attention_weights = attention_weights.softmax(-1)
        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams, self.num_levels)

        # TODO: use if else to switch between dynamic conv and weighted sum 

        attention_weights = self.weight_dropout(attention_weights.sigmoid()) * mask
        output = output * attention_weights
        # [bs, embed_dims, num_query]
        output = output.sum(-1).sum(-1).sum(-1)
        # change to [bs, num_query, embed_dims]
        output = output.permute(0, 2, 1)

        # [B, M, radar_dim], [B, M,]
        radar_feats, radar_mask = radar_feats[:, :, :-1], radar_feats[:, :, -1]

        radar_xy = radar_feats[:, :, :2]
        ref_xy = reference_points[:,:,:2]
        radar_feats = radar_feats[:, :, 2:]

        pad_xy = torch.ones_like(radar_xy) * 1000.0
        
        radar_xy = radar_xy + (1.0 - radar_mask.unsqueeze(dim=-1).type(torch.float)) * (pad_xy)
        
        # [B, num_query, M]
        ref_radar_dist = -1.0 * torch.cdist(ref_xy, radar_xy)

        # [B, num_query, topk]
        _value, indices = torch.topk(ref_radar_dist, self.radar_topk)
        
        # [B, num_query, M]
        radar_mask = radar_mask.unsqueeze(dim=1).repeat(1, num_query, 1)

        # [B, num_query, topk]
        top_mask = torch.gather(radar_mask, 2, indices)

        # [B, num_query, M, radar_dim]
        radar_feats = radar_feats.unsqueeze(dim=1).repeat(1, num_query, 1, 1)
        radar_dim = radar_feats.size(-1)
        # [B, num_query, topk, radar_dim]
        indices_pad = indices.unsqueeze(dim=-1).repeat(1, 1, 1, radar_dim)

        # [B, num_query, topk, radar_dim]
        radar_feats_topk = torch.gather(radar_feats, dim=2, 
                        index=indices_pad, sparse_grad=False)
        
        attention_weights_radar = self.attention_weights_radar(query).view(
            bs, num_query, self.radar_topk)

        # [B, num_query, topk]
        attention_weights_radar = self.weight_dropout(attention_weights_radar.sigmoid()) * top_mask
        # [B, num_query, topk, radar_dim]
        radar_out = radar_feats_topk * attention_weights_radar.unsqueeze(dim=-1)
        # [bs, num_query, radar_dim]
        radar_out = radar_out.sum(dim=2)
        radar_out = self.radar_out_proj(radar_out)
        B, num_query, radar_dim = radar_out.size()

        ones_ = radar_out.new_ones((B, num_query, 1))
        ones_ = self.drop_radar(ones_)
        radar_out = radar_out * ones_

        ones_ = output.new_ones((B, num_query, 1))
        ones_ = self.drop_img(ones_)
        output = output * ones_
        
        # (bs, num_query, radar_dims + embed_dim)
        output = torch.cat((output, radar_out), dim=-1)
        output = self.output_proj(output)

        # change to (num_query, bs, embed_dims)
        output = output.permute(1, 0, 2)
        # (num_query, bs, embed_dims)
        return self.dropout(output) + inp_residual + self.pos_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points_3d = reference_points.clone()
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    num_points = 1
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat_flip = torch.flip(feat, [-1])
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, int(num_query/10), 10, 2)
        # offsets_lvl = offsets[:, lvl, :, :, :].reshape(B*N, num_query, num_points , 2)
        # reference_points_cam_lvl = reference_points_cam_lvl + offsets_lvl
        # (B* num_cam, num_query, 2), (B, N, C, H, W)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  num_points, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask



@ATTENTION.register_module()
class Detr3DCamRadarSparseDynamicCrossAtten(BaseModule):
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
                 radar_dims=64, 
                 radar_topk=3, 
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 drop_img=0.0, 
                 drop_radar=0.0, 
                 use_level_cam_embed=False,
                 norm_cfg=None,
                 init_cfg=None, 
                 batch_first=False, 
                 **kwargs, 
                 ):
        super(Detr3DCamRadarSparseDynamicCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.drop_img = nn.Dropout(drop_img)
        self.drop_radar = nn.Dropout(drop_radar)
        self.pc_range = pc_range
        self.radar_dims = radar_dims

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

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_level_cam_embed = use_level_cam_embed
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        
        self.query2radar_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.radar_dims), 
        )
        self.radar_topk = radar_topk

        self.radar_out_proj = nn.Sequential(
            nn.Linear(self.radar_dims, self.radar_dims), 
        )

        self.radar_value_proj = nn.Linear(self.radar_dims, self.radar_dims)

        self.attn_div = math.sqrt(radar_dims * 1.0)

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims+radar_dims, embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )
        
        if self.use_level_cam_embed:
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.embed_dims, self.num_levels))
            self.cam_embeds = nn.Parameter(
                torch.Tensor(self.embed_dims, self.num_cams))

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_level_cam_embed:
            normal_(self.level_embeds)
            normal_(self.cam_embeds)
    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                radar_feats=None,  # [B, M, K]
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
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

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        if self.use_level_cam_embed:
            level_embeds = self.level_embeds.view(1, self.embed_dims, 1, 1, 1, self.num_levels)
            cam_embeds = self.cam_embeds.view(1, self.embed_dims, 1, self.num_cams, 1, 1)
            output = output + level_embeds + cam_embeds

        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams * self.num_levels)
        # attention_weights = attention_weights.softmax(-1)
        # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams, self.num_levels)

        # TODO: use if else to switch between dynamic conv and weighted sum 

        attention_weights = self.weight_dropout(attention_weights.sigmoid()) * mask
        output = output * attention_weights
        # [bs, embed_dims, num_query]
        output = output.sum(-1).sum(-1).sum(-1)
        # change to [bs, num_query, embed_dims]
        output = output.permute(0, 2, 1)

        # [B, M, radar_dim], [B, M,]
        radar_feats, radar_mask = radar_feats[:, :, :-1], radar_feats[:, :, -1]

        radar_xy = radar_feats[:, :, :2]
        ref_xy = reference_points[:,:,:2]
        radar_feats = radar_feats[:, :, 2:]

        pad_xy = torch.ones_like(radar_xy) * 1000.0
        
        radar_xy = radar_xy + (1.0 - radar_mask.unsqueeze(dim=-1).type(torch.float)) * (pad_xy)
        
        # [B, num_query, M]
        ref_radar_dist = -1.0 * torch.cdist(ref_xy, radar_xy)

        # [B, num_query, topk]
        _value, indices = torch.topk(ref_radar_dist, self.radar_topk)
        
        # [B, num_query, M]
        radar_mask = radar_mask.unsqueeze(dim=1).repeat(1, num_query, 1)

        # [B, num_query, topk]
        top_mask = torch.gather(radar_mask, 2, indices)

        # [B, num_query, M, radar_dim]
        radar_feats = radar_feats.unsqueeze(dim=1).repeat(1, num_query, 1, 1)
        radar_dim = radar_feats.size(-1)
        # [B, num_query, topk, radar_dim]
        indices_pad = indices.unsqueeze(dim=-1).repeat(1, 1, 1, radar_dim)

        # [B, num_query, topk, radar_dim]
        radar_feats_topk = torch.gather(radar_feats, dim=2, 
                        index=indices_pad, sparse_grad=False)
        
        # [B, num_query, radar_dim]
        radar_query = self.query2radar_embed(query)
        # [B, num_query, topk, radar_dim]
        radar_feats_in = self.radar_out_proj(radar_feats_topk)
        # [B, num_query, topk, radar_dim]
        radar_feats_value = self.radar_value_proj(radar_feats_topk)

        # matmul: [B, num_query, topk]
        radar_attn_weights = torch.matmul(radar_feats_in, radar_query.unsqueeze(dim=-1)).squeeze(dim=-1)

        radar_attn_weights = torch.softmax(radar_attn_weights / self.attn_div, dim=-1)

        # [B, num_query, topk, radar_dim] 
        radar_out = (radar_attn_weights * top_mask).unsqueeze(dim=-1) * radar_feats_value
        radar_out = radar_out.sum(dim=2)
        
        B, num_query, radar_dim = radar_out.size()

        ones_ = radar_out.new_ones((B, num_query, 1))
        ones_ = self.drop_radar(ones_)
        radar_out = radar_out * ones_

        ones_ = output.new_ones((B, num_query, 1))
        ones_ = self.drop_img(ones_)
        output = output * ones_

        # (bs, num_query, radar_dims + embed_dim)
        output = torch.cat((output, radar_out), dim=-1)
        output = self.output_proj(output)

        # change to (num_query, bs, embed_dims)
        output = output.permute(1, 0, 2)
        # (num_query, bs, embed_dims)
        return self.dropout(output) + inp_residual + self.pos_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)