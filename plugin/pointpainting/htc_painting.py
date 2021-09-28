# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN


@DETECTORS.register_module()
class HybridTaskCascadePainting(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(HybridTaskCascadePainting, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        self.simple_test(data['img'], data['img_metas'], data['points'], data['sweep_points'])

        losses = 0

        outputs = dict(
            loss=losses, num_samples=len(data['img_metas']))

        return outputs

    def simple_test(self, img, img_metas, points, sweep_points, proposals=None, rescale=False):
        """Test without augmentation."""

        pts_metas = img_metas[0]['pts_metas']
        img_metas[0].pop('pts_metas')
        img_metas[0]['flip'] = False
        #print(len(img_metas), img_metas[0].keys())
        B, num_cams, C, H, W = img.size()
        img = img.view(B*num_cams, C, H, W)
        mul_img_metas = [{} for i in range(num_cams)]
        for key in img_metas[0]:
            if isinstance(img_metas[0][key], list):
                for i in range(num_cams):
                    mul_img_metas[i][key] = img_metas[0][key][i]
            else:
                for i in range(num_cams):
                    mul_img_metas[i][key] = img_metas[0][key]

        #assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, mul_img_metas)
        else:
            proposal_list = proposals

        results = self.roi_head.simple_test(
            x, proposal_list, mul_img_metas, points, sweep_points, pts_metas, rescale=rescale)

        return results