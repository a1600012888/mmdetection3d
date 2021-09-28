_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/nus-3d.py',
    './htc_r50_fpn_model.py'
]
dataset_type = 'NuScenesDatasetv2'
data_root = 'data/nuscenes/'

plugin=True
plugin_dir='plugin/pointpainting/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

#model = dict(
#    type='HybridTaskCascadePainting',
#)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFilev2',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFilesv2'),
    dict(
        type='LoadPointsFromMultiSweepsv2',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=False,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler),
    #dict(
    #    type='GlobalRotScaleTrans',
    #    rot_range=[-0.3925, 0.3925],
    #    scale_ratio_range=[0.95, 1.05],
    #    translation_std=[0, 0, 0]),
    #dict(
    #    type='RandomFlip3D',
    #    sync_2d=False,
    #    flip_ratio_bev_horizontal=0.5,
    #    flip_ratio_bev_vertical=0.5),
    #dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    #dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    #dict(type='ObjectNameFilter', classes=class_names),
    #dict(type='PointShuffle'),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3Dv2', size_divisor=32),
    dict(type='DefaultFormatBundle3Dv2', class_names=class_names, with_gt=False),
    dict(type='Collect3Dv2', keys=['points', 'img', 'sweep_points'])
]
test_pipeline = [
    #type=dataset_type,
    dict(
        type='LoadPointsFromFilev2',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='LoadPointsFromMultiSweepsv2',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=False,
        remove_close=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
    #revise
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=False,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        #type='CBGSDataset',
        #dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
     #),
    val=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality))


optimizer = dict(
    type='AdamW',
    lr=0,
    weight_decay=0.01)

# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[1])

'''
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
'''
total_epochs = 1
evaluation = dict(interval=1, pipeline=eval_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=1)

find_unused_parameters = False

workflow = [('val', 1)]

load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_20e_nuim/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth'