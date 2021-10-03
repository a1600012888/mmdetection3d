_base_ = [
    '../_base/base_plus_3f.py'
]

model = dict(
    fix_feats=False,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.1, random_drop=0.1),
)

find_unused_parameters = True

fp16 = dict(loss_scale='dynamic')
load_from='work_dirs/track/lidar_velo/rdar_cam_xywlzh_12ep_fix_radar_attn_notanh_detach_ft12ep+moreaug_12ep/latest.pth'
#load_from='work_dirs/track/v2/fp16/base3f_mem/latest.pth'
