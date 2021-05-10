
from tqdm import tqdm
from nuscenes.utils import splits
import json
import os
from nuscenes.nuscenes import NuScenes, NuScenesExplorer

SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test'}
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']

def get_scene_name(sample_token, nusc):
    sample_data = nusc.get('sample', sample_token)
    scene_data = nusc.get('scene', sample_data['scene_token'])
    scene_name = scene_data['name']

    return scene_name


def split_depth_meta(meta, nusc):
    train_scenes = splits.train
    val_scenes = splits.val

    train_meta = []
    val_meta = []

    for m in tqdm(meta):
        sample_token = m['sample_token']
        scene_name = get_scene_name(sample_token, nusc)

        if scene_name in train_scenes:
            train_meta.append(m)
        elif scene_name in val_scenes:
            val_meta.append(m)
        else:
            print('scene {} not in train or val'.format(scene_name))
    
    return train_meta, val_meta

def transform_depth(root_dir='/public/MARS/datasets/nuScenes/depth_maps'):
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    
    meta_path = os.path.join(root_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    train_meta, val_meta = split_depth_meta(meta, nusc)

    save_dir='/public/MARS/datasets/nuScenes-SF/depth_meta'
    train_meta_path = os.path.join(save_dir, 'meta_train.json')
    val_meta_path = os.path.join(save_dir, 'meta_val.json')

    with open(train_meta_path, 'w') as f:
        json.dump(train_meta, f)

    with open(val_meta_path, 'w') as f:
        json.dump(val_meta, f)


if __name__ == '__main__':
    transform_depth(root_dir='/public/MARS/datasets/nuScenes/depth_maps')