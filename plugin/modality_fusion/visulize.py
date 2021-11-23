from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import json
import os

from pyquaternion import Quaternion
from nuscenes import NuScenes
from collections import OrderedDict

results_path = 'xxx.json'
with open(results_path) as f:
    data = json.load(f)
    results_dict = data['results']

    new_results_dict = {}

    for key, item in results_dict.items():
        new_item = []
        for _box_dict in item:
            if  'detection_name' in _box_dict:
                score=_box_dict['detection_score']
                if score < 0.25:
                    continue
                new_box = Box(
                    center=_box_dict['translation'],
                    size=_box_dict['size'],
                    orientation=Quaternion(_box_dict['rotation']),
                    score=_box_dict['detection_score'],
                    velocity=_box_dict['velocity'] + [0],
                    name=_box_dict['detection_name'],
                    token=_box_dict['sample_token'])
                new_item.append(new_box)
        new_results_dict[key] = new_item

if isinstance(box, DetectionBox):
        box = Box(box.translation, box.size, Quaternion(box.rotation))
        cam_name = 'CAM_FRONT'
 
        sd_record = nusc.get('sample_data', sample_data_token)

        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)