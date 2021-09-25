_base_ = [
    '../_base/radar_base_5f_noradar.py'
]

model = dict(
    fix_feats=True,
)

find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
#load_from = 'work_dirs/models/backbone_neck.pth'
load_from='work_dirs/track/v3/res50_cam_velo_lf_ep12_ft12ep/latest.pth'
