from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import numpy as np
import os
import json

SPLITS = {'val': 'v1.0-trainval-val', 'train': 'v1.0-trainval', 'test': 'v1.0-test'}
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']
LidarName = 'LIDAR_TOP'
# preserve: 'token' -> sample_token, 'scene_token', data.cam_token -> cam_token

def convert(split='val', data_path='data/nuscenes/', save_path='data/nuscenes/depth_maps'):
    nusc = NuScenes(
        version=SPLITS[split], dataroot=data_path, verbose=True)
    nusc_exp = NuScenesExplorer(nusc)

    save_dir = os.path.join(save_path, split)
    if not os.path.isdir(save_dir):
        pass

    ret = []

    for sample in nusc.sample:
        
        sample_token = sample['token']
        print(sample_token, len(ret))

        lidar_token = sample['data'][LidarName]        

        for cam_name in CamNames:
            cam_token = sample['data'][cam_name]
            depth_map_path = sample_token + cam_name + '.pts'
            depth_map_path = os.path.join(save_dir, 'depth_data', depth_map_path)
            img_path = nusc.get_sample_data_path(cam_token)

            data_info = {}
            data_info['sample_token'] = sample_token
            data_info['lidar_token'] = lidar_token
            data_info['depth_path'] = depth_map_path
            data_info['img_path'] = img_path
            
            ret.append(data_info)
            continue
            points, coloring, im = nusc_exp.map_pointcloud_to_image(lidar_token, cam_token)

            float_x_cords = points[0] # < 1600
            float_y_cords = points[1] # < 900
            float_depth = coloring # 

            point_with_depth = np.stack([float_x_cords, float_y_cords, float_depth], axis=-1)
            np.save(depth_map_path, point_with_depth)

            
            #nusc.render_pointcloud_in_image(sample_token, camera_channel='CAM_FRONT', out_path='./render.png', verbose=False)
    
    meta_file_path = os.path.join(save_dir, 'meta.json')
    with open(meta_file_path, 'w') as f:
        json.dump(ret, f)

def generate_depth_map(points, coloring, img):
    '''
    points: [3, n]:  x, y, z.  
    coloring: depth of shape [n,]
    img: np.ndarray of shape [900, 1600, 3]
    '''
    h, w, c = img.shape
    if c is not 1:
        coloring = coloring[:, np.newaxis].repeat(c, -1)

    # float cord
    x_cords = points[0]  # max  < 1600
    y_cords = points[1] # < 900

    # -> int cord
    x_cords = x_cords.astype(np.int)
    y_cords = y_cords.astype(np.int)
    coloring = coloring.astype(np.int)

    img[y_cords, x_cords] = coloring




if __name__ == '__main__':
    convert()
