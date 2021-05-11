from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import os
import json
from mmdet3d.datasets.pipelines import Compose
import numpy as np
import copy

@DATASETS.register_module()
class NuscSpatialTemp(Dataset):

    CLASSES=2
    def __init__(self, sf_path='/public/MARS/datasets/nuScenes-SF/trainval',
                img_path='data/nuscenes/'
                pipeline=None, training=True, **kwargs):

        self.sf_path = sf_path
        self.img_path = img_path

        self.training = training
        if self.training:
            self.meta_path = '/public/MARS/datasets/nuScenes-SF/meta/spatial_temp_train.json'
            self.depth_meta_path = '/public/MARS/datasets/nuScenes-SF/depth_meta/meta_train.json'
            self.sf_meta_path = '/public/MARS/datasets/nuScenes-SF/meta/meta_sf_train.json'
        else:
            self.meta_path = '/public/MARS/datasets/nuScenes-SF/meta/spatial_temp_val.json'
            self.depth_meta_path = '/public/MARS/datasets/nuScenes-SF/depth_meta/meta_val.json'
            self.sf_meta_path = '/public/MARS/datasets/nuScenes-SF/meta/meta_sf_val.json'

        self.data_infos = self.load_annotations()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        # not important
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self):

        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        with open(self.depth_meta_path, 'r') as f:
            self.depth_meta = json.load(f)

        with open(self.sf_meta_path, 'r') as f:
            self.sf_meta = json.load(f)
        
        self.imgpath2paths = {}

        for depth_info in self.depth_meta:
            token = depth_info['sample_token']
            depth_path = depth_info['points_path']
            sf_path = os.path.join(self.sf_path, self.sf_meta[token]['points_path'])
            img_path = os.path.join(self.img_path, self.sf_meta[token]['img_path'])
            
            tmp = {'token': token, 'depth_path': depth_path, 
                    'sf_path': sf_path, 'img_path': img_path}
            
            self.imgpath2paths[self.sf_meta[token]['img_path']] = tmp

        data_infos = []

        for data_info in self.meta:
            # now, prev, next
            file_names = []
            depth_paths = []
            sf_paths = []

            for time_key in ['prev', 'now', 'next']:
                img_names = data_info[time_key]
                for img_name in img_names:
                    paths_info = self.imgpath2paths[img_name]
                    depth_path = paths_info['depth_path'] + 'npy' # +'.npy'
                    file_names.append(paths_info['img_path'])
                    depth_paths.append(depth_path)
                    sf_paths.append(paths_info['sf_path'])
            
            tmp = {'img_info': {'filename': file_names, 'img_prefix': None},
                'npy_info': {'depth_paths': depth_paths, 
                            'sf_paths': sf_paths}
                }

            data_infos.append(tmp)

        return data_infos

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img_prefix'] = None
        data_info['flip'] = None
        data_info['flip_direction'] = None
        return data_info

    def __len__(self,):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.get_data_info(idx)

        data = self.pipeline(data_info)

        return data

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None):
        # [(loss.item(), twenty_acc.item(), ten_acc.item(), five_acc.item(), one_acc.item())]
        # print(results)


        abs_diff = [res[0] for res in results]
        abs_rel = [res[1] for res in results]
        sq_rel = [res[2] for res in results]
        rmse = [res[3] for res in results]
        rmse_log = [res[4] for res in results]
        losses = [res[5] for res in results]

        num_batch = len(losses)

        abs_diff = sum(abs_diff) / num_batch
        abs_rel = sum(abs_rel) / num_batch
        sq_rel = sum(sq_rel) / num_batch
        rmse = sum(rmse) / num_batch
        rmse_log = sum(rmse_log) / num_batch
        loss = sum(losses) / num_batch



        #print(results, loss)
        ret_dict = {'loss': loss,
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log
                     }

        return ret_dict


    def format_results(self, outputs, **kwargs):
        ret_dict = self.evaluate(outputs)

        print(ret_dict)
