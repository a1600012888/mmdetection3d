from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/chenxy/mmdetection3d/data/nuscenes', verbose=True)

cali_sensors = nusc.calibrated_sensor

#print(nusc.sensor)
print(len(cali_sensors))

#for cali_sensor in cali_sensors:
#    if cali_sensor['sensor_token'] == 'dc8b396651c05aedbb9cdaae573bb567':
#        print(cali_sensor)
