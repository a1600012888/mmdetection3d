import argparse

from tools.visualization.nusc_explorer import NuScenesMars, NuScenesExplorerMars

import mmcv

from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.tracking.data_classes import TrackingBox

from nuscenes.utils.data_classes import Box


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
        version='v1.0-trainval', dataroot='../../data/nuscenes')
    nusc_exp = NuScenesExplorerMars(nusc)
    
    samples = nusc.sample

    pred_boxes, meta = load_prediction(args.result_path, 300, TrackingBox,
                                       verbose=True)
    
    sample_tokens = pred_boxes.sample_tokens[args.star_ind: args.end_ind]

    box_list_dict = {}
    for sample_token in sample_tokens:
        box_list = pred_boxes[sample_token]
        tmp_list = []

        for track_box in tmp_list:
            box = Box(track_box.translation, track_box.size, track_box.rotation, 
            label=track_box.tracking_id, score=track_box.tracking_score)


    boxes = pred_boxes[args.star_ind: args.end_ind]



