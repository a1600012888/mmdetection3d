from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import os
import json
from mmdet3d.datasets.pipelines import Compose
import numpy as np
import copy

@DATASETS.register_module()
class NuscDepthDataset(Dataset):

    CLASSES=2
    def __init__(self, data_path='data/nuscenes/depth_maps/train/detpth_map', 
                pipeline=None, training=True, **kwargs):
        self.depth_root = os.path.join(data_path, 'depth_data')
        self.meta_path = os.path.join(data_path, 'meta.json')
        self.training = training
        if training:
            #self.data_infos = self.load_annotations()[:-20000]
            self.data_infos = self.load_annotations()[:-20000]
        else:
            #self.data_infos = self.load_annotations()[-20000:]
            self.data_infos = self.load_annotations()[-20000:]
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        # not important
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self):

        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        data_infos = []

        for data_info in self.meta:
            img_path = data_info['img_path']
            depth_path = data_info['depth_path']+'.npy' # depth_path+'.npy'
            
            tmp = {'img_info':{'filename':img_path}, 
                'npy_info': {'filename': depth_path}}

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
        losses = [res[0] for res in results]
        acc_20 = [res[1] for res in results]
        acc_10 = [res[2] for res in results]
        acc_5 = [res[3] for res in results]
        acc_1 = [res[4] for res in results]

        loss = sum(losses) / len(losses)
        acc_20 = sum(acc_20) / len(acc_20)
        acc_10 = sum(acc_10) / len(acc_10)
        acc_5 = sum(acc_5) / len(acc_5)
        acc_1 = sum(acc_1) / len(acc_1)


        #print(results, loss)
        ret_dict = {'loss': loss, 'acc_20': acc_20, 'acc_10':acc_10, 'acc_5':acc_5, 'acc_1':acc_1 }

        return ret_dict
        

