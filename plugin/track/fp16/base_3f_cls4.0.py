_base_ = [
    '../_base/base_3f_cls4.0.py'
]

model = dict(
    fix_feats=False,
)

find_unused_parameters = True

fp16 = dict(loss_scale='dynamic')
load_from='/home/ubuntu/projects/detr_det/mmdetection3d/work_dirs/track/v2/fp16/base3f_mem/latest.pth'
