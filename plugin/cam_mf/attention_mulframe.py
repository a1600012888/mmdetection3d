
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_, xavier_normal_


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


@ATTENTION.register_module()
class Detr3DCamCrossAttenMulFrame(BaseModule):
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
                 num_points=4,
                 num_cams=6,
                 num_frames=2,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 use_frame_offsets=False,
                 use_frame_embed=False,
                 pos_embed_dims=16,
                 norm_cfg=None,
                 init_cfg=None):
        super(Detr3DCamCrossAttenMulFrame, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.pc_range = pc_range

        self.pos_embed_dims = pos_embed_dims
        self.pos_embed_subnet = nn.Sequential(
            nn.Linear(3*pos_embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

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
        self.num_frames = num_frames
        self.use_frame_embed = use_frame_embed
        self.use_frame_offsets = use_frame_offsets

        self.attention_weights = nn.Linear(
            embed_dims, num_cams*num_heads*num_points*num_levels*num_frames)
        self.offsets = nn.Linear(embed_dims, num_heads*num_points*3*num_frames)
        if self.use_frame_offsets:
            self.frame_offsets = nn.linear(embed_dims, 3 * (num_frames-1))

        self.value_proj = nn.Linear(embed_dims,
                                    embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.frame_fuse = nn.Sequential(
            nn.Linear(embed_dims*num_frames, embed_dims), 
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
        )

        if self.use_frame_embed:
            self.frame_embeds = nn.parameter(
                torch.Tensor(self.embed_dims * self.num_frames))
        
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.offsets, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.frame_fuse, distribution='uniform', bias=0.)
        if self.use_frame_embed:
            xavier_init(self.frame_embeds, distribution='uniform', bias=0.)

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
        for _f, frame_feat in enumerate(value):
            frame_list = []
            for lvl, feat in enumerate(frame_feat):
                _B, _N, _C, _H, _W = feat.shape
                # feat: [B, N, C, H, W] => [B, N, H, W, C]
                value_flat = feat.permute(0, 1, 3, 4, 2)

                value_flat_proj = self.value_proj(value_flat)

                # [B, N, H, W, num_heads, dim_per_head]
                value_flat_proj = value_flat_proj.view(_B, _N, _H, _W, self.num_heads, -1)

                frame_list.append(value_flat_proj)
            value_proj_list.append(frame_list)
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        # [bs, num_query, num_cams*num_heads*num_points*num_levels*num_frame]
        attention_weights = self.attention_weights(query)

        # [B, num_query, num_heads*num_points*3]
        # shape [B, 1, num_query, 1, num_heads, num_points, 3]
        offsets_norm = query.new_tensor([0.5, 0.5, 0.5]).view(1, 1, 1, 1, 1, 1, 3, 1) / self.num_points
        offsets = self.offsets(query).view(bs, 1, num_query, 1, self.num_heads, self.num_points, 3, self.num_frames) * offsets_norm  # or not?
        # shape [B, num_cam, num_query, num_level, num_heads, num_points, 3]
        offsets = offsets.repeat(1, self.num_cams, 1, self.num_levels, 1, 1, 1, 1)

        if self.use_frame_offsets:
            frame_offsets = self.frame_offsets(query).view(bs, num_query, 3, self.num_frames - 1)
            frame_offsets_list = [frame_offsets[..., i] for i in range(self.num_frames - 1)]
            frame_offsets_list.append(None)
        else:
            frame_offsets_list = [None] * self.num_frames

        attention_weights = attention_weights.view(bs, num_query, -1, self.num_frames)
        # note. split will keep dim
        # Tuple[(bs, num_query, -1, 1)]
        attn_weights_list = attention_weights.split(dim=-1, split_size=1)

        offsets_list = offsets.split(dim=-1, split_size=1)
        offsets_list = [_a.squeeze(dim=-1) for _a in offsets_list]
        
        frame_out_list = []
        for frame_off, attn_weights, off in zip(frame_offsets_list, attn_weights_list, offsets_list):

            if frame_off is not None:
                ref_points = inverse_sigmoid(reference_points)
                ref_points = ref_points + frame_off
                ref_points = ref_points.sigmoid().clone()
            else:
                ref_points = reference_points
            # [B, num_query, num_heads, dim_per_head/1, num_cam, num_points, num_level]
            # lidar2img may not change in two frames. so we use the same
            output, mask = feature_sampling(value_proj_list, ref_points, off, self.pc_range, kwargs['img_metas'])
            output = torch.nan_to_num(output)
            mask = torch.nan_to_num(mask)

            # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams * self.num_levels)
            # attention_weights = attention_weights.softmax(-1)
            # attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams, self.num_levels)

            # [bs, num_query, num_cams*num_heads*num_points*num_levels]
            # attention_weights = (attention_weights / torch.sqrt(output.new_tensor(self.embed_dims*1.0))).softmax(-1)
            # attention_weights = attention_weights.softmax(-1)
            attention_weights = attn_weights.sigmoid()

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
            output = torch.flatten(output, 2, 3)
            # [bs, num_query, C] => [num_query, bs, C]
            output = output.permute(1, 0, 2)

            frame_out_list.append(output)
        
        mul_frame_out = torch.cat(frame_out_list)
        if self.use_frame_embed:
            frame_embeds = self.frame_embeds.view(1, 1, -1)
            mul_frame_out = mul_frame_out + frame_embeds

        output = self.frame_fuse(mul_frame_out)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)

        pos_encoding = self.pos_embed_subnet(postional_embedding(reference_points, self.pos_embed_dims))
        pos_encoding = pos_encoding.permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_encoding


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
    reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp(reference_points_cam[..., 2:3], min=1e-5)
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
