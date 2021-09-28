import torch

#checkpoint1 = torch.load('/public/MARS/models/surrdet/points_model/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth')

checkpoint1 = torch.load('/home/chenxy/mmdetection3d/work_dirs/centerpoint_pointonly_0075voxel_q6_dataaug_fade_step_38e/epoch_2.pth')
state_dict1 = checkpoint1['state_dict']

checkpoint2 = torch.load('/public/MARS/models/surrdet/image_models/detrcam_3ddet_fcos3d_pre_24ep.pth')
state_dict2 = checkpoint2['state_dict']


state_dict2.update(state_dict1)
merged_state_dict = state_dict2

save_checkpoint = {'state_dict':merged_state_dict }

'''
for name in list(merged_state_dict.keys()):
    if name.startswith('pts_bbox_head'):
        merged_state_dict.pop(name)
'''

torch.save(save_checkpoint, '/public/MARS/models/surrdet/pts_img_models/img_3473_0075voxel_q6_epoch_38_6012.pth') 

'''
checkpoint = torch.load('/home/chenxy/mmdetection3d/work_dirs/res101_prefcos3d_centerpoint_pre_02pillar_step_20e/epoch_20.pth')
state_dict = checkpoint['state_dict']
print(len(state_dict))
for name in list(state_dict.keys()):
    if name.startswith('pts'):
        if (name.find('reference_points') == -1) and (name.find('query_embedding') == -1):
            state_dict.pop(name)

for name in state_dict:
    print(name)

save_checkpoint = {'state_dict': state_dict}
torch.save(save_checkpoint, '/public/MARS/models/surrdet/pts_img_models/res101_02pillar_fusion_erase_epoch20.pth')
'''