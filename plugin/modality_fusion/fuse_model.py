import torch

checkpoint1 = torch.load('/public/MARS/models/surrdet/points_model/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth')
state_dict1 = checkpoint1['state_dict']

checkpoint2 = torch.load('/public/MARS/models/surrdet/image_models/fcos3d.pth')
state_dict2 = checkpoint2['state_dict']


state_dict1.update(state_dict2)
merged_state_dict = state_dict1

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, '/public/MARS/models/surrdet/pts_img_models/fcos3d_centerpoint_02pillar.pth') 