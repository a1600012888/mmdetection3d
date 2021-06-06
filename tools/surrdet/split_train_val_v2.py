
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

def split_sf_meta(meta, nusc):
    train_scenes = splits.train
    val_scenes = splits.val

    train_meta = {}
    val_meta = {}

    for m, value in tqdm(meta.items()):
        sample_token = nusc.get('sample_data', m)['sample_token']
        scene_name = get_scene_name(sample_token, nusc)

        if scene_name in train_scenes:
            train_meta[m] = value
        elif scene_name in val_scenes:
            val_meta[m] = value
        else:
            print('scene {} not in train or val'.format(scene_name))
    return train_meta, val_meta

def transform_sceneflow(root_dir='/public/MARS/datasets/nuScenes-SF/meta'):
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    
    meta_path = os.path.join(root_dir, 'meta_sf.json')

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    train_meta, val_meta = split_sf_meta(meta, nusc)

    train_meta_path = os.path.join(root_dir, 'meta_sf_train.json')
    val_meta_path = os.path.join(root_dir, 'meta_sf_val.json')

    with open(train_meta_path, 'w') as f:
        json.dump(train_meta, f)

    with open(val_meta_path, 'w') as f:
        json.dump(val_meta, f)


def get_cam_filename_dict(token_dict, nusc):
    ret_dict = {}
    for name, token in token_dict.items():
        ret_dict[name] = []
        sample_data = nusc.get('sample', token)
        for cam_name in CamNames:
            cam_token = sample_data['data'][cam_name]
            filename = nusc.get('sample_data', cam_token)['filename']
            ret_dict[name].append(filename)
    
    return ret_dict


def get_temporal_spatial_data(save_dir='/public/MARS/datasets/nuScenes-SF/meta'):
    split='train'
    data_path='data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)

    train_scenes = splits.train
    val_scenes = splits.val

    train_meta = []
    val_meta = []
    
    for sample in tqdm(nusc.sample):
        ret = {
            'now': {'filename': [], 'cam_token': []}, 
            'prev': {'filename': [], 'cam_token': []},  
            'next': {'filename': [], 'cam_token': []}, 
        }
        sample_token = sample['token']
        prev_token = sample['prev']
        next_token = sample['next']
        sample_data = nusc.get('sample', sample_token)
        if prev_token == '' or next_token == '':
            continue
        
        data_complete = True
        for cam_name in CamNames:
            cam_token = sample_data['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            filename = cam_data['filename']
            ret['now']['filename'].append(filename)
            ret['now']['cam_token'].append(cam_token)

            prev_cam_token = cam_data['prev']
            if prev_cam_token == '':
                data_complete = False
                break
            prev_cam_data = nusc.get('sample_data', prev_cam_token)
            filename = prev_cam_data['filename']
            ret['prev']['filename'].append(filename)
            ret['prev']['cam_token'].append(prev_cam_token)

            next_cam_token = cam_data['next']
            if next_cam_token == '':
                data_complete = False
                break
            next_cam_data = nusc.get('sample_data', next_cam_token)
            filename = next_cam_data['filename']
            ret['next']['filename'].append(filename)
            ret['next']['cam_token'].append(next_cam_token)

        if not data_complete:
            continue
        scene_name = get_scene_name(sample_token, nusc)
        cam_filename_dict = ret

        if scene_name in train_scenes:
            train_meta.append(cam_filename_dict)
        else:
            val_meta.append(cam_filename_dict)
    
    train_meta_path = os.path.join(save_dir, 'spatial_temp_train_v2.json')
    val_meta_path = os.path.join(save_dir, 'spatial_temp_val_v2.json')
    print('len of train_meta: ', len(train_meta))
    with open(train_meta_path, 'w') as f:
        json.dump(train_meta, f)
    with open(val_meta_path, 'w') as f:
        json.dump(val_meta, f)
        

def merge_depth_sf(depth_meta_path, 
                sf_meta_path, save_path):
    
    with open(depth_meta_path, 'r') as f:
        depth_meta = json.load(f)

    with open(sf_meta_path, 'r') as f:
        sf_meta = json.load(f)
    
    split = 'train'
    data_path = 'data/nuscenes/'
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    
    imgpath2paths = {}
    for depth_info in depth_meta:
        sample_token = depth_info['sample_token']
    
        depth_path = depth_info['depth_path']
        img_path = depth_info['img_path']
        cam_name = img_path.split('/')[-2]

        cam_token = nusc.get('sample', sample_token)['data'][cam_name]
        sf_next_path = sf_meta[cam_token]['points_path_next']
        sf_prev_path = sf_meta[cam_token]['points_path_prev']
        img_path = sf_meta[cam_token]['img_path'] # use this version of img path

        tmp = {'token': cam_token, 'depth_path': depth_path, 
                'cam_name': cam_name, 'sf_next_path': sf_next_path, 
                'sf_prev_path': sf_prev_path, 'img_path': img_path}
        imgpath2paths[img_path] = tmp

    with open(save_path, 'w') as f:
        json.dump(imgpath2paths, f)


if __name__ == '__main__':
    #transform_depth(root_dir='/public/MARS/datasets/nuScenes/depth_maps')
    #get_temporal_spatial_data()
    #transform_sceneflow('/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta')

    merge_depth_sf(depth_meta_path = '/public/MARS/datasets/nuScenes-SF/depth_meta/meta_val.json', 
                sf_meta_path = '/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/meta_sf_val.json', 
                save_path='/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/spatial_temp_merged_path_val.json')

