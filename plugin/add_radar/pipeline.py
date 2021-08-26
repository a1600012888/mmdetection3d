import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from nuscenes.utils.data_classes import RadarPointCloud

@PIPELINES.register_module()
class LoadRadarPoints(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 file_client_args=dict(backend='disk'),
                 max_num=150,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

        self.total_num = 0
        self.cnt = 0
    

    def _cal_stat(self, delta):
        self.total_num += delta
        self.cnt += 1

        print('avg {:.2f} - cnt: {}'.format(self.total_num / self.cnt, self.cnt))

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']

        points_sweep_list = []
        for key, sweep in radars_dict.items():
            points_sweep = self._load_points(sweep['data_path'])
            points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
            
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep_list.append(points_sweep)
        
        points = np.concatenate(points_sweep_list, axis=0)
        
        points = points[:, self.use_dim]
        #print(points.shape)
        #self._cal_stat(points.shape[0])

        points[:, 0:1] = (points[:, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        points[:, 1:2] = (points[:, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        points[:, 2:3] = (points[:, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.max_num > 0:
            points, mask = self._pad_or_drop(points)
        else:
            mask = np.ones((points.shape[0], points.shape[1]), 
                        dtype=points.dtype)
        
        points = np.concatenate((points, mask), axis=-1)
        

        results['radar'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep_list.append(points_sweep)
        
        points = np.concatenate(points_sweep_list, axis=0)
        
        points = points[:, self.use_dim]

        points[:, 0:1] = (points[:, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        points[:, 1:2] = (points[:, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        points[:, 2:3] = (points[:, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.max_num > 0:
            points, mask = self._pad_or_drop(points)
        else:
            mask = np.ones((points.shape[0], points.shape[1]), 
                        dtype=points.dtype)
        
        points = np.concatenate((points, mask), axis=-1)

        results['radar'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'