_base_ = [
    '../_base/base_3f.py'
]

model = dict(
    fix_feats=False,
)

find_unused_parameters = True

fp16 = dict(loss_scale='dynamic')
load_from='work_dirs/track/v2/fp16/base3f_mem/latest.pth'
