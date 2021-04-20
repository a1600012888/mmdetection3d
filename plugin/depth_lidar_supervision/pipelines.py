from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from typing import Tuple, List, Union
import numpy as np
import torch

@PIPELINES.register_module()
class LoadDepthImage(object):

    _render_types = ['naive']

    def __init__(self, img_size=(480, 892), render_type='naive'):
        '''
        img_size: List or Tuple with two elemets: h, w
        '''

        assert render_type in self._render_types, 'render_type:{} is not supported!'.format(render_type)
        self.render_type = render_type
        self.img_size = img_size # ()
        self.ylim, self.xlim = img_size
    

    def naive_depth_render(self, points, depth_map):
        '''
        for float cord, use its int version
        '''
        x_cords = points[:, 0] * self.xlim / 1600.0
        y_cords = points[:, 1] * self.ylim / 900.0
        depth = points[:, 2]

        #print('debug', x_cords.max(), y_cords.max(), depth_map.shape)
        x_cords = x_cords.astype(np.int)
        y_cords = y_cords.astype(np.int)
        
        # first y, then x
        #print(depth_map.shape, )
        depth_map[y_cords, x_cords] = points[:,2]

        return depth_map

    def __call__(self, results):

        npy_file_path = results['npy_info']['filename']

        points = np.load(npy_file_path) # of shape [N, 3]: x, y, depth

        depth_map = np.zeros(self.img_size)
        if depth_map.ndim == 2:
            #depth_map = depth_map[:, :, np.newaxis]
            pass
        
        if self.render_type == 'naive':
            depth_map = self.naive_depth_render(points, depth_map)
        
        # 900x1600 => 1600x900
        # depth_map = depth_map.transpose(1,0)
        #results['seg_fields'] = torch.tensor(depth_map)
        results['seg_fields'] = ['depth_map']
        results['depth_map'] = depth_map
        #depth_map[:,:, np.newaxis]
        #print('debbug', results['depth_map'].shape, results['img'].shape, type(results['img']), '\n')

        
        return results

    



        