import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv import ConfigDict, deprecated_api_warning
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init, constant_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import Transformer
from .attention import inverse_sigmoid

def postional_embedding(x, embed_dims=16):
    '''
    x: [..., 3]
    '''
    cord = 2. * x - 1.0

    freq_bands = (2. ** torch.linspace(0.0, embed_dims // 2 -1, embed_dims // 2, device=x.device)) * 3.1416

    enc_x = torch.cat(
        [torch.sin(cord[..., [0]] * freq_bands),
        torch.cos(cord[..., [0]] * freq_bands)], dim=-1,
    )
    enc_y = torch.cat(
        [torch.sin(cord[..., [1]] * freq_bands),
        torch.cos(cord[..., [1]] * freq_bands)], dim=-1,
    )
    enc_z = torch.cat(
        [torch.sin(cord[..., [2]] * freq_bands),
        torch.cos(cord[..., [2]] * freq_bands)], dim=-1,
    )

    enc = torch.cat([enc_x, enc_y, enc_z], dim=-1)

    return enc


def feature_sampling_3D_offsets(mlvl_feats, reference_points, offsets, pc_range):
    '''
    Args
        
    '''

    reference_points = inverse_sigmoid(reference_points)
    B, num_query, _ = reference_points.shape
    reference_points = reference_points.view(B, 1, num_query, 1, 1, 1, _)
    B, N, _, num_levels, num_heads, num_points, _ = offsets.shape

    # shape (B, N, num_query, num_levels, num_heads, num_points, 3)
    reference_points = reference_points.repeat(1, N, 1, num_levels, num_heads, num_points, 1)
    #print(reference_points.size(), offsets.size())
    reference_points = reference_points + offsets
    reference_points = reference_points.sigmoid() # inplace op below;
    reference_points = reference_points.clone()

    # shape (B, N, num_query, num_levels, num_heads, num_points, 2)
    reference_points_rel = reference_points[..., 0:2]
    
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points_rel[..., 0] = reference_points[..., 0] / pc_range[3]
    reference_points_rel[..., 1] = reference_points[..., 1] / pc_range[4]
    
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, H, W, _num_heads, dim_per_head, = feat.size()
        # shape (B, N, num_heads, dim_per_head, H, W)
        feat = feat.permute(0, 1, 4, 5, 2, 3)
        # shape (B*N*num_heads, dim_per_head, H, W)
        feat = torch.flatten(feat, start_dim=0, end_dim=2)
        # reference_points_rel_lvl (B, N, num_qeury, num_heads, num_points, 2)
        reference_points_rel_lvl = reference_points_rel[:, :, :, lvl, ...]
        # shape (B, N, num_heads, num_query, num_points, 2)
        reference_points_rel_lvl = reference_points_rel_lvl.permute(0, 1, 3, 2, 4, 5)
        # shape (B*N*num_heads, num_query, num_points, 2)
        reference_points_rel_lvl = torch.flatten(reference_points_rel_lvl, 0, 2)
        # shape (B*N*num_heads, dim_per_head, num_query, num_points)
        sampled_feat = F.grid_sample(feat, reference_points_rel_lvl, mode='bilinear', 
                                        padding_mode='zeros', align_corners=False)
        # [B, N, num_heads, dim_per_head, num_query, num_points]
        sampled_feat = sampled_feat.unflatten(0, (B, N, num_heads))
        # [B, num_query, num_heads, dim_per_head, N, num_points]
        sampled_feat = sampled_feat.permute(0, 4, 2, 3, 1, 5)
        sampled_feats.append(sampled_feat)
    # sampled_feats shape [B, num_query, num_heads, dim_per_head, N, num_points, num_levels]
    sampled_feats = torch.stack(sampled_feats, -1)

    return sampled_feats


@ATTENTION.register_module()
class Detr3DCamCrossAttenPointOffset(BaseModule):
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
                 pos_embed_dims=16,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(Detr3DCamCrossAttenPointOffset, self).__init__(init_cfg)
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

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed
        self.pos_embed_dims = pos_embed_dims
        self.pos_embed_subnet = nn.Sequential(
            nn.Linear(3*pos_embed_dims, embed_dims),
            nn.LayerNorm(embed_dims), 
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims), 
        )
        self.pts_attention_weights = nn.Linear(embed_dims, 
                                            num_heads*num_levels*num_points)

        self.pts_output_proj = nn.Linear(embed_dims, embed_dims)

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )
        self.offsets = nn.Linear(embed_dims, num_heads*num_points*3)

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.pts_attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.offsets, distribution='uniform', bias=0.)

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
                img_feats=None,
                pts_feats=None,
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

        # value projection
        value_proj_list = []
        for lvl, feat in enumerate(pts_feats):
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

        pts_attention_weights =  self.pts_attention_weights(query)

        # [B, num_query, num_heads*num_points*3]
        # shape [B, 1, num_query, 1, num_heads, num_points, 3]
        offsets = self.offsets(query).view(bs, 1, num_query, 1, self.num_heads, self.num_points, 3) / self.num_points  # or not?
        # shape [B, num_cam, num_query, num_level, num_heads, num_points, 3]
        offsets = offsets.repeat(1, 1, 1, self.num_levels, 1, 1, 1)

        # pts_output shape [B, num_query, num_heads, dim_per_head, N, num_points, num_levels]
        pts_output= feature_sampling_3D_offsets(
            value_proj_list, reference_points, offsets, self.pc_range)
        pts_output = torch.nan_to_num(pts_output)
        
        # pts_attention_weights shape [bs, num_query, num_heads*num_points*num_levels]
        pts_attention_weights = self.weight_dropout(pts_attention_weights.sigmoid())
        pts_attention_weights = pts_attention_weights.view(
            bs, num_query, self.num_heads, 1, 1, self.num_points, self.num_levels)
        # pts_output shape (B, num_query, num_heads, dim_per_head, N, num_points, num_levels)
        pts_output = pts_output * pts_attention_weights
        # pts_output shape (B, num_query, num_heads, dim_per_head)
        pts_output = pts_output.sum(-1).sum(-1).sum(-1)
        # pts_output shape (B, num_query, embed_dims)
        pts_output = torch.flatten(pts_output, 2, 3)
        # pts_output shape (num_query, B, embed_dims)
        pts_output = pts_output.permute(1, 0, 2)

        pts_output = self.pts_output_proj(pts_output)

        # shape (B, num_query, embed_dims)
        pos_encoding = self.pos_embed_subnet(postional_embedding(reference_points, self.pos_embed_dims))
        # shape (num_query, B, embed_dims)
        pos_encoding = pos_encoding.permute(1, 0, 2)

        return self.dropout(pts_output) + inp_residual + pos_encoding
