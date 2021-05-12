from tqdm import tqdm
import json
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np

SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test'}
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']


def quat_trans2matrix(quant, translation):
    quant_matrix = Quaternion(quant).rotation_matrix
    translation = np.array(translation)

    # shape [3, 4]
    matrix = np.concatenate([quant_matrix, translation[:, np.newaxis]], axis=-1)
    last_line = np.array([0.0, 0.0, 0.0, 1.0])

    # shape [4, 4]
    matrix_full = np.concatenate([matrix, last_line[np.newaxis, ]], axis=0)

    return matrix_full


def get_pose_intrinsic(save_path='/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic.json'):

    split = 'train'
    data_path = 'data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    samples = nusc.sample

    cam_token2cam_intext = {}

    for sample in tqdm(samples):
        for cam_name in CamNames:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

            pose_matrix = quat_trans2matrix(ego_pose['rotation'], ego_pose['translation'])
            cam_pose = quat_trans2matrix(cam_cs['rotation'], cam_cs['translation'])

            cam_pose_world = np.matmul(pose_matrix, cam_pose)

            ret = {'pose': cam_pose_world.tolist()}
            ret['intrinsic'] = cam_cs['camera_intrinsic']

            cam_token2cam_intext[cam_token] = ret
    
    with open(save_path, 'w') as f:
        json.dump(cam_token2cam_intext, f)
    

if __name__ == '__main__':
    get_pose_intrinsic()
