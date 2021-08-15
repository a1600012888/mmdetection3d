import torch

#checkpoint1 = torch.load('/public/MARS/models/surrdet/points_model/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth')
checkpoint1 = torch.load('/home/chenxy/mmdetection3d/work_dirs/centerpoint_pointonly_02pillar_q6_dataaug_fade_step2_40e/epoch_6.pth')
state_dict1 = checkpoint1['state_dict']

checkpoint2 = torch.load('/public/MARS/models/surrdet/image_models/fcos3d.pth')
state_dict2 = checkpoint2['state_dict']


state_dict1.update(state_dict2)
merged_state_dict = state_dict1

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, '/public/MARS/models/surrdet/pts_img_models/fcos3d_02pillar_q6_fade_40e.pth') 