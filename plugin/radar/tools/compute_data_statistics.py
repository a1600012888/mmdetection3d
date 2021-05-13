import json
import os
import numpy as np

def get_sf_sparsity():
    point_dir = '/public/MARS/datasets/nuScenes-SF/trainval/'
    file_names = [a for a in os.listdir(point_dir) if a.endswith('.npy')]

    total_num = 0
    for i, file_name in enumerate(file_names):
        point_path = os.path.join(point_dir, file_name)
        points = np.load(point_path)
        num = points.shape[1]
        total_num += num

        if i == 100000:
            print('avg points for sf: {}'.format(total_num / i))
            break
        

if __name__ == '__main__':
    get_sf_sparsity()