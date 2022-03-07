import argparse

from tools.visualization.nusc_explorer import NuScenesMars, NuScenesExplorerMars, load_results_json
from tools.visualization.video_maker import make_sensor_pred_videos, make_sensor_videos

import mmcv

from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.tracking.data_classes import TrackingBox

from nuscenes.utils.data_classes import Box
from copy import deepcopy
import os


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='MarsDet. Show 3D Det/Track results')

    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument(
        '--show-gt',
        action='store_true',
        help='Whether to show gt results')
    parser.add_argument('--out', help='output result images/jif in directory', 
                        default='./work_dirs/visualization/tmp')
    parser.add_argument('--start_ind', type=int, default=0)
    parser.add_argument('--end_ind', type=int, default=100)

    args = parser.parse_args()

    nusc = NuScenesMars(
        version='v1.0-trainval', dataroot='data/nuscenes')
    nusc_exp = NuScenesExplorerMars(nusc)
    
    samples = nusc.sample

    results_dict = load_results_json(args.result_path)
    # index must start 0
    selected_keys = list(results_dict.keys())[:100]

    gt_dir = os.path.join(args.out, 'gt')
    pred_dir = os.path.join(args.out, 'pred')
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    
    sensors = ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK']
    for sensor in sensors:
        sensor_tokens = []
        for sample_token in selected_keys:
            sensor_tokens.append(nusc.get('sample', sample_token)['data'][sensor])
        
        make_sensor_pred_videos(
            sensor_tokens, selected_keys,
            deepcopy(results_dict), nusc_exp,
            out_dir=os.path.join(pred_dir, sensor), fps=2)
        if args.show_gt:
            make_sensor_videos(sensor_tokens, nusc_exp,
                               out_dir=os.path.join(gt_dir, sensor),
                               fps=2)


