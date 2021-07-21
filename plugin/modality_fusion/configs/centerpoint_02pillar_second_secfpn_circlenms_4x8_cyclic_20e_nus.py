#https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint/
# centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py

_base_ = ['./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))