from os import times
import torch
import torch.nn as nn

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.utils.grid import GridMask
from mmdet3d.core.bbox.coders import build_bbox_coder
from ..structures import Instances
from .qim import build_qim
from .memory_bank import build_memory_bank
from mmdet.models import build_loss
from copy import deepcopy
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.core.bbox.util import normalize_bbox, denormalize_bbox


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1

    def update_fix_label(self, track_instances: Instances, old_class_scores):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] >= self.filter_score_thresh:
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]


@DETECTORS.register_module()
class Detr3DCamTracker(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 embed_dims=256,
                 num_query=300,
                 num_classes=7,
                 bbox_coder=dict(
                    type='DETRTrack3DCoder',
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    max_num=300,
                    num_classes=7),
                 qim_args=dict(
                     qim_type='QIMBase',
                     merger_dropout=0, update_query_pos=False,
                     fp_ratio=0.3, random_drop=0.1),
                 mem_cfg=dict(
                     memory_bank_type='MemoryBank',
                     memory_bank_score_thresh=0.0,
                     memory_bank_len=4,
                 ),
                 fix_feats=False,
                 score_thresh=0.2,
                 filter_score_thresh=0.1,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 loss_cfg=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(Detr3DCamTracker,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = embed_dims
        self.num_query = num_query
        self.fix_feats = fix_feats
        if self.fix_feats:
            self.img_backbone.eval()
            self.img_neck.eval()
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)
        self.mem_bank_len = 10
        self.memory_bank = None
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=5)

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.memory_bank = build_memory_bank(
            args=mem_cfg,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if self.fix_feats:
            with torch.no_grad():
                img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = self.extract_img_feat(img, img_metas)
        return (img_feats, None)

    def _targets_to_instances(self, gt_bboxes_3d=None,
                              gt_labels_3d=None, instance_inds=None,
                              img_shape=(1, 1,)):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = gt_bboxes_3d
        gt_instances.labels = gt_labels_3d
        gt_instances.obj_ids = instance_inds
        return gt_instances

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(
                            query[..., :dim // 2])

        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device)

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len),
            dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            (len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embedding.weight.device)

    def _copy_tracks_for_loss(self, tgt_instances):

        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)

        track_instances.save_period = deepcopy(tgt_instances.save_period)
        return track_instances.to(self.query_embedding.weight.device)

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=('img', 'points'))
    def _forward_single(self, points, img, img_metas, track_instances):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        '''
        B, num_cam, _, H, W = img.shape
        img_feats, pts_feats = self.extract_feat(points, img=img,
                                                 img_metas=img_metas)
        # img_feats = [a.clone() for a in img_feats]

        # output_classes: [num_dec, B, num_query, num_classes]
        # query_feats: [B, num_query, embed_dim]

        output_classes, output_coords, \
            query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, track_instances.query,
                track_instances.ref_pts, img_metas,)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the mather.
        track_instances_list = [self._copy_tracks_for_loss(track_instances) for i in range(nb_dec-1)]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]

        # track_instances.query = torch.cat((track_instances.query[:, :self.embed_dims // 2],
        #   query_feats[0]), dim=1)

        track_instances_list.append(track_instances)
        for i in range(nb_dec):
            track_instances = track_instances_list[i]
            #track_scores = output_classes[i, 0, :].sigmoid().max(dim=-1).values

            # no loss for this term?
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]

            out['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1)))

        # TODO calc_loss_for_track_scores
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            # self.criterion.calc_loss_for_track_scores(track_instances)

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        out['track_instances'] = out_track_instances
        return out

    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      instance_inds=None,
                      gt_bboxes_ignore=None,
                      timestamp=None,
                      ):
        """Forward training function.
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bs = img.size(0)
        num_frame = img.size(1)
        track_instances = self._generate_empty_tracks()

        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = instance_inds[0][i]
            gt_instances_list.append(gt_instances)

        # TODO init criterion
        self.criterion.initialize_for_single_clip(gt_instances_list)

        # for bs 1
        lidar2img = img_metas[0]['lidar2img']  # [T, num_cam]
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points]
            img_single = torch.stack([img_[i] for img_ in img], dim=0)

            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]['lidar2img'] = lidar2img[i]

            frame_res = self._forward_single(points_single, img_single,
                                             img_metas_single, track_instances)
            track_instances = frame_res['track_instances']

        outputs = self.criterion.losses_dict
        return outputs

    def _inference_single(self, points, img, img_metas, track_instances):
        '''
        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        '''
        B, num_cam, _, H, W = img.shape
        img_feats, pts_feats = self.extract_feat(points, img=img,
                                                 img_metas=img_metas)
        img_feats = [a.clone() for a in img_feats]

        # output_classes: [num_dec, B, num_query, num_classes]
        # query_feats: [B, num_query, embed_dim]

        output_classes, output_coords, \
            query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, track_instances.query,
                track_instances.ref_pts, img_metas,)

        out = {'pred_logits': output_classes[-1],
               'pred_boxes': output_coords[-1],
               'ref_pts': last_ref_pts}

        # TODO: Why no max?
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # track_scores = output_classes[-1, 0, :, 0].sigmoid()

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]

        self.track_base.update(track_instances)
        #self.track_base.update_fix_label(track_instances, old_class_scores)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        out['track_instances'] = out_track_instances
        return out

    def forward_test(self,
                     points=None,
                     img=None,
                     img_metas=None,
                     timestamp=1e6,
                     **kwargs,
                     ):
        """Forward test function.
        only support bs=1, single-gpu, num_frame=1 test
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bs = img.size(0)
        num_frame = img.size(1)

        timestamp = timestamp[0]
        if self.test_track_instances is None:
            track_instances = self._generate_empty_tracks()
            self.test_track_instances = track_instances
            self.timestamp = timestamp[0]
        # TODO: use scene tokens?
        if timestamp[0] - self.timestamp > 10:
            track_instances = self._generate_empty_tracks()
        else:
            track_instances = self.test_track_instances
        self.timestamp = timestamp[-1]

        # for bs 1;
        lidar2img = img_metas[0]['lidar2img']  # [T, num_cam]
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points]
            img_single = torch.stack([img_[i] for img_ in img], dim=0)

            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]['lidar2img'] = lidar2img[i]

            frame_res = self._inference_single(points_single, img_single,
                                               img_metas_single,
                                               track_instances)
            track_instances = frame_res['track_instances']

        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances))
        self.test_track_instances = track_instances

        results = self._active_instances2results(active_instances, img_metas)
        return results

    def _active_instances2results(self, active_instances, img_metas):
        '''
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        '''
        # filter out sleep querys
        active_idxes = (active_instances.scores >= self.track_base.filter_score_thresh)
        active_instances = active_instances[active_idxes]
        if active_instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=active_instances.pred_logits,
            bbox_preds=active_instances.pred_boxes,
            track_scores=active_instances.scores,
            obj_idxes=active_instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]['box_type_3d'][0](bboxes, 9)
        labels = bboxes_dict['labels']
        scores = bboxes_dict['scores']

        track_scores = bboxes_dict['track_scores']
        obj_idxes = bboxes_dict['obj_idxes']
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
        )

        return [result_dict]

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.
        The function implementation process is as follows:
            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.
        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.
        Returns:
            dict: Returned bboxes consists of the following keys:
                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        pts_bbox = self.aug_test_pts(img_feats, img_metas, rescale)
        bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

