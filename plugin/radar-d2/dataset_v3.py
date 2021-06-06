from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import os
import json
from mmdet3d.datasets.pipelines import Compose
import numpy as np
import copy
from collections import OrderedDict

@DATASETS.register_module()
class NuscSpatialTempV3(Dataset):

    CLASSES=2

    def __init__(self, sf_path='/public/MARS/datasets/nuScenes-SF/trainval-camera',
                img_path='data/nuscenes/', 
                pose_path='/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic_v3.json', 
                pipeline=None, training=True, **kwargs):

        self.sf_path = sf_path
        self.img_path = img_path
        self.pose_path = pose_path
        self.training = training
        if self.training:
            self.depth_pred_path = '/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/sf_inp_train_meta.json'
            self.meta_path = '/public/MARS/datasets/nuScenes-SF/meta/spatial_temp_train_v2.json'
            self.merged_file_path = '/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/spatial_temp_merged_path_train.json'
        else:
            self.depth_pred_path = '/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/sf_inp_val_meta.json' # change need
            self.meta_path = '/public/MARS/datasets/nuScenes-SF/meta/spatial_temp_val_v2.json'
            self.merged_file_path = '/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/spatial_temp_merged_path_val.json'

        self.data_infos = self.load_annotations()#[:200]
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        # not important
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self):

        with open(self.depth_pred_path, 'r') as f:
            self.depth_pred_dict = json.load(f)

        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        with open(self.merged_file_path, 'r') as f:
            self.imgpath2paths = json.load(f)

        with open(self.pose_path, 'r') as f:
            self.token2pose = json.load(f)

        data_infos = []

        for data_info in self.meta:
            # now, prev, next
            file_names = []
            depth_paths = []
            sf_next_paths = []
            sf_prev_paths = []
            cam_intrinsics = []
            cam_poses = []
            timestamp = []

            for time_key in ['prev', 'now', 'next']:
                temp_dict = data_info[time_key]
                img_names = temp_dict['filename']
                cam_tokens = temp_dict['cam_token']
                if time_key == 'next':
                    img_name = img_names[-1]
                    depth_pred_path = self.depth_pred_dict[img_name]
                for i, img_name in enumerate(img_names):
                    
                    if time_key == 'now':
                        paths_info = self.imgpath2paths[img_name]
                        cam_token = paths_info['token']
                        cam_intrinsic = self.token2pose[cam_token]['intrinsic']
                        cam_pose = self.token2pose[cam_token]['pose']
                        cam_intrinsics.append(cam_intrinsic)
                        cam_poses.append(cam_pose)

                        depth_path = paths_info['depth_path'] + '.npy' # +'.npy'
                        file_names.append(os.path.join(self.img_path, paths_info['img_path']))
                        depth_paths.append(depth_path)
                        if paths_info['sf_next_path'] is not None:
                            sf_next_paths.append(os.path.join(self.sf_path, paths_info['sf_next_path']))
                        else:
                            sf_next_paths.append(None)
                        if paths_info['sf_prev_path'] is not None:
                            sf_prev_paths.append(os.path.join(self.sf_path, paths_info['sf_prev_path']))
                        else:
                            sf_prev_paths.append(None)
                        timestamp.append(self.token2pose[cam_token]['timestamp'] * 1e-6)
                    else:
                        depth_paths.append(None)
                        sf_next_paths.append(None)
                        sf_prev_paths.append(None)
                        
                        file_names.append(os.path.join(self.img_path, img_name))
                        cam_token = cam_tokens[i]
                        cam_intrinsic = self.token2pose[cam_token]['intrinsic']
                        cam_pose = self.token2pose[cam_token]['pose']
                        cam_intrinsics.append(cam_intrinsic)
                        cam_poses.append(cam_pose)
                        timestamp.append(self.token2pose[cam_token]['timestamp'] * 1e-6) 
            
            prev_t, now_t, next_t = timestamp[:6], timestamp[6:12], timestamp[12:]
            now2next = [ne-no for ne, no in zip(next_t, now_t)]
            now2prev = [pre-no for pre, no in zip(prev_t, now_t)]
            
            tmp = {'img_info': {'filenames': file_names, 'img_prefix': None, 
                                'filename': file_names[0]},
                'npy_info': {'depth_paths': depth_paths, 
                            'sf_next_paths': sf_next_paths, 
                            'sf_prev_paths': sf_prev_paths, 
                            'depth_pred_path': depth_pred_path}, 
                'cam_intrinsic': np.array(cam_intrinsics).astype(np.float32),  # [4,4] 
                'cam_pose': np.array(cam_poses).astype(np.float32),  # [3, 3]
                'now2next_time': np.array(now2next).astype(np.float32), 
                'now2prev_time': np.array(now2prev).astype(np.float32), 
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

        ret_keys = ['abs_rel', 'loss', 'epe', 
            'epe_rel', 'thres_0.1', 'thres_0.3',
            'thres_0.5', 'temp_rec_loss']
        
        ret_dict = OrderedDict()
        for i, ret_key in enumerate(ret_keys):
            temp_val = [res[i] for res in results]
            num = len(temp_val)
            val = sum(temp_val) / num
            ret_dict[ret_key] = val

        return ret_dict


    def format_results(self, outputs, **kwargs):
        ret_dict = self.evaluate(outputs)

        print(ret_dict)
