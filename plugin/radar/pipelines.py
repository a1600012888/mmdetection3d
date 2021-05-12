from tools.surrdet.get_scene_flow_v2 import points_to_ann_cord
from IPython.terminal.embed import embed
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from typing import Tuple, List, Union
import numpy as np
import torch
import mmcv

@PIPELINES.register_module()
class LoadDepthImages(object):

    _render_types = ['naive']

    def __init__(self, img_size=(480, 892), render_type='naive'):
        '''
        img_size: List or Tuple with two elemets: h, w
        '''

        assert render_type in self._render_types, 'render_type:{} is not supported!'.format(render_type)
        self.render_type = render_type
        self.img_size = img_size # ()
        self.ylim, self.xlim = img_size

    def sort_points(self, points):
        '''
        sort the points accroding to their depth in descending order
        '''
        depth = points[:, 2]
        idx = np.argsort(depth) # ascending order
        idx = idx[::-1]

        new_points = points[idx]

        return new_points

    def naive_depth_render(self, points, depth_map):
        '''
        for float cord, use its int version
        '''
        points = self.sort_points(points)

        x_cords = points[:, 0] * self.xlim / 1600.0
        y_cords = points[:, 1] * self.ylim / 900.0
        depth = points[:, 2]
        depth = np.clip(depth, a_min=1e-5, a_max=99999)

        #print('debug', depth[:10].mean(), depth[10:100].mean(), depth[-100:].mean())
        #print('debug', x_cords.max(), y_cords.max(), depth_map.shape)
        x_cords = x_cords.astype(np.int)
        y_cords = y_cords.astype(np.int)

        # first y, then x
        #print(depth_map.shape, )
        depth_map[y_cords, x_cords] = points[:,2]

        return depth_map

    def __call__(self, results):

        npy_file_paths = results['npy_info']['depth_paths']
        
        i = 0
        results['seg_fields'] = []
        for npy_file_path in npy_file_paths:
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

            results['seg_fields'].append('depth_map{}'.format(i))
            results['depth_map{}'.format(i)] = depth_map
            i += 1
            #depth_map[:,:, np.newaxis]
            #print('debbug', results['depth_map'].shape, results['img'].shape, type(results['img']), '\n')

        return results


@PIPELINES.register_module()
class LoadSceneFlows(object):

    _render_types = ['naive']

    def __init__(self, img_size=(480, 892), render_type='naive'):
        '''
        img_size: List or Tuple with two elemets: h, w
        '''

        assert render_type in self._render_types, 'render_type:{} is not supported!'.format(render_type)
        self.render_type = render_type
        self.img_size = img_size # ()
        self.ylim, self.xlim = img_size

    def sort_points(self, points):
        '''
        sort the points accroding to their depth in descending order
        '''
        depth = points[:, 2]
        idx = np.argsort(depth) # ascending order
        idx = idx[::-1]

        new_points = points[idx]

        return new_points

    def naive_sceneflow_render(self, points, sf_map, valid_mask):
        '''
        for float cord, use its int version
        '''
        points = points.transpose()
        
        points = self.sort_points(points)

        # filter out noisy static points
        scene_flow = points[:, 3:]
        speed = np.linalg.norm(scene_flow, axis=-1)
        moving_mask = (speed > 0.2)
        
        static_points = points[np.logical_not(moving_mask)]
        moving_points = points[moving_mask]

        static_points[:, 3:] = static_points[:, 3:] * 0

        points = np.concatenate([moving_points, static_points], axis=0)

        x_cords = points[:, 0] * self.xlim / 1600.0
        y_cords = points[:, 1] * self.ylim / 900.0
        x_cords = x_cords.astype(np.int)
        y_cords = y_cords.astype(np.int)

        scene_flow = np.clip(points[:, 3:], a_min=-100, a_max=100)
        
        sf_map[y_cords, x_cords, :] = scene_flow
        valid_mask[y_cords, x_cords] = 1

        sf_map = np.concatenate([sf_map, valid_mask], axis=-1)
        return sf_map

    def __call__(self, results):

        npy_file_paths = results['npy_info']['sf_paths']
        
        i = 0
        for npy_file_path in npy_file_paths:
            if npy_file_path is None:
                sf_map = np.zeros((*self.img_size, 4))
                results['seg_fields'].append('sf_map{}'.format(i))
                results['sf_map{}'.format(i)] = sf_map
                i += 1
                continue

            points = np.load(npy_file_path)  # of shape [N, 3]: x, y, depth
            sf_map = np.zeros((*self.img_size, 3))
            valid_mask = np.zeros((*self.img_size, 1))
            
            if self.render_type == 'naive':
                sf_map = self.naive_sceneflow_render(points, sf_map, valid_mask) # [H,W,4]

            # 900x1600 => 1600x900
            # depth_map = depth_map.transpose(1,0)
            #results['seg_fields'] = torch.tensor(depth_map)

            results['seg_fields'].append('sf_map{}'.format(i))
            results['sf_map{}'.format(i)] = sf_map
            i += 1
            
        return results

@PIPELINES.register_module()
class ResizeDepthImage(object):

    def __init__(self, scale:float = 1/4, interpolation='bilinear'):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, results):
        depth_map = results['depth_map'] #
        new_depth_map = mmcv.imrescale(depth_map, self.scale, interpolation=self.interpolation)

        results['depth_map'] = new_depth_map
        return results



@PIPELINES.register_module()
class LoadImageFromFiles(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                to_float32=False,
                color_type='color',
                file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        filenames = results['img_info']['filenames']

        i = 0
        results['img_fields'] = []
        for filename in filenames:
    
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            
            results['img{}'.format(i)] = img
            results['img_fields'].append('img{}'.format(i))
            i += 1

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str