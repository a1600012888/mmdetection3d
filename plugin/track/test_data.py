import torch
import numpy as np


def _test():

    file_client_args = dict(backend='disk')
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]

    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    dataset_type = 'NuScenesTrackDataset'
    data_root = 'data/nuscenes/'
    class_names = [
        'car', 'truck', 'bus', 'trailer', 
        'motorcycle', 'bicycle', 'pedestrian',
    ]
    input_modality = dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False)
    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args),
        dict(type='LoadMultiViewImageFromFiles'),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=1,
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=file_client_args,
            pad_empty_sweeps=True,
            remove_close=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=class_names),
        dict(type='Normalize3D', **img_norm_cfg),
        dict(type='Pad3D', size_divisor=32)]

    train_pipeline_post = [
        dict(type='FormatBundle3DTrack'),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img'])
    ]
    
    data = dict(
        samples_per_gpu=1,
        workers_per_gpu=4,
        train=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'track_infos_train.pkl',
                pipeline_single=train_pipeline,
                pipeline_post=train_pipeline_post,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                use_valid_flag=True,
                # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
                # and box_type_3d='Depth' in sunrgbd and scannet dataset.
                box_type_3d='LiDAR'),)

    from plugin.track.pipeline import FormatBundle3DTrack
    from mmdet3d.datasets import build_dataset

    dataset = build_dataset(data['train'])

    from IPython import embed
    embed()


if __name__ == '__main__':
    _test()
    
