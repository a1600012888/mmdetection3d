import os
import random
import numpy as np
CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']

def random_sample(data_path, ):
    file_names = os.listdir(data_path)
    file_names = [f for f in file_names if f.endswith('.npy')]

    sample_file_names = random.choices(file_names, k=1000)

    ret = [os.path.join(data_path, f) for f in sample_file_names]

    return ret

def analyze(data_path):

    random_sample_paths = random_sample(data_path)

    totoal_point_counts = 0
    counts_dict = {v:0 for v in CamNames}
    div_dict = {v:0 for v in CamNames}
    for path in random_sample_paths:
        pts = np.load(path)

        totoal_point_counts += pts.shape[0]
        count = pts.shape[0]
        for cam_name in CamNames:
            if path.find(cam_name) != -1:
                print(path, cam_name)
                counts_dict[cam_name] += count
                div_dict[cam_name] += 1
    
    print('avg point coumts = {}'.format(totoal_point_counts/len(random_sample_paths)))
    for cam_name in CamNames:
        print('Number of avg points for {} is {}.'.format(cam_name, counts_dict[cam_name]/div_dict[cam_name], div_dict[cam_name]))



if __name__ == '__main__':

    analyze('data/nuscenes/depth_maps/train/depth_data')




