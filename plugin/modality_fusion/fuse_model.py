import torch
'''
checkpoint1 = torch.load('checkpoint/dd3d_nuscenes.pth')
model = checkpoint1['model']
filtered_model = {}
for name in model:
    if name.startswith('backbone.bottom_up'):
        new_name = 'img_backbone' + name[18:]
        filtered_model[new_name] = model[name]
        print(name)
        print(new_name)

save_checkpoint = {'state_dict':filtered_model}
torch.save(save_checkpoint, 'checkpoint/dd3d_nuscenes_backbone.pth')
'''

checkpoint1 = torch.load('/public/MARS/models/surrdet/image_models/detrcam_3ddet_fcos3d_pre_24ep.pth')
#checkpoint1 = torch.load('/public/MARS/models/surrdet/img_radar_models/img_radar_3504_epoch24.pth')
state_dict1 = checkpoint1['state_dict']

#checkpoint2 = torch.load('/home/chenxy/mmdetection3d/work_dirs/02pillar_vismap_reduce4_q6_dataaug_fade_step_38e/epoch_2.pth')
#checkpoint2 = torch.load('/home/chenxy/mmdetection3d/work_dirs/centerpoint_pointonly_01voxel_q6_dataaug_fade_step_38e/epoch_2.pth')
checkpoint2 = torch.load('/home/chenxy/mmdetection3d/work_dirs/01voxel_reduce1_beam10_q6_dataaug_fade_step_38e/epoch_2.pth')
state_dict2 = checkpoint2['state_dict']

state_dict2.update(state_dict1)
merged_state_dict = state_dict2

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, '/public/MARS/models/surrdet/pts_img_models/img_3473_01voxel_reduce1_beam10_q6_epoch_38_1710_imgdecoder.pth')


