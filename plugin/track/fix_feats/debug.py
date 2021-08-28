_base_ = [
    '../_base/base_3f.py'
]

model = dict(
    fix_feats=True,
)

find_unused_parameters = True
load_from = 'work_dirs/models/backbone_neck.pth'
