_base_ = [
    '../_base/base_plus_5f.py'
]

model = dict(
    fix_feats=True,
)

find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
#load_from = 'work_dirs/models/backbone_neck.pth'
load_from='work_dirs/track/lidar_velo/rdar_cam_xywlzh_12ep_fix_radar_attn_notanh_detach_ft12ep+12ep_smallaug/latest.pth'
