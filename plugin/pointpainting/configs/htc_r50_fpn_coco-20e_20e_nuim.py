_base_ = './htc_r50_fpn_coco-20e_1x_nuim.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20

load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_20e_nuim/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth'