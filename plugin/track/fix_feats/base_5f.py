_base_ = [
    '../_base/base_5f.py'
]

model = dict(
    fix_feats=True,
)

find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
#load_from = 'work_dirs/models/backbone_neck.pth'
load_from='work_dirs/track/v2/fp16/base3f_mem_24ep_test_truck/latest.pth'
