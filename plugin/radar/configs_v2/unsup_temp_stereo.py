

# Set plugin = True
plugin = True
plugin_dir = 'plugin/radar/'


#model = dict(
#    type='UNet',
#    in_channels=3,
#    out_channels=1,
#    base_channels=16,
#    num_stages=5,
#    strides=(1, 1, 1, 1, 1),
#    enc_num_convs=(2, 2, 2, 2, 2),
#    dec_num_convs=(2, 2, 2, 2),
#    downsamples=(True, True, True, True),
#    norm_cfg=dict(type='BN'),
#    act_cfg=dict(type='ReLU'),
#    upsample_cfg=dict(type='InterpConv'),
#    )
model = dict(
    type='SpatialTempNet',
    depth_net_cfg={'version': '1A', },
    sf_net_cfg=None,
    scale_depth=False,
    scale_depth_for_temp=False,
    depth_supervision_ratio=-1,
    depth_smoothing=1e-3,
    motion_smoothing=0,
    motion_sparse=0,
    sf_consis=0,
    depth_consis=0,
    rgb_consis=1.0,
    stereo_rgb_consis=0.2,
    loss_decay=0.25,
)


file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
#mean=[58.395, 57.12, 57.375], std=[123.675, 116.28, 103.53], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFiles'), # filenames = results['img_info']['filenames']； results['img{}'.format(i)] = img
    dict(
        type='Resize',
        img_scale=(384, 224), # w, h; note after reading is (h=900, w=1600)
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg), # results.get('img_fields', ['img'])
    dict(type='LoadDepthImages', img_size=(224, 384), render_type='naive'), # results['seg_fields']
    dict(type='LoadSceneFlows', img_size=(224, 384), render_type='naive'), # results['seg_fields']
    dict(type='ImageToTensor', keys=['img{}'.format(i) for i in range(18)]),
    dict(type='Collect', keys=['img{}'.format(i) for i in range(18)] + \
                            ['depth_map{}'.format(i) for i in range(18)] + \
                            ['sf_map{}'.format(i) for i in range(18)] + \
                            ['cam_intrinsic', 'cam_pose']),

]
val_pipeline = [
    dict(type='LoadImageFromFiles'), # filenames = results['img_info']['filenames']； results['img{}'.format(i)] = img
    dict(
        type='Resize',
        img_scale=(384, 224), # w, h; note after reading is (h=900, w=1600)
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg), # results.get('img_fields', ['img'])
    dict(type='LoadDepthImages', img_size=(224, 384), render_type='naive'), # results['seg_fields']
    dict(type='LoadSceneFlows', img_size=(224, 384), render_type='naive'), # results['seg_fields']
    dict(type='ImageToTensor', keys=['img{}'.format(i) for i in range(18)]),
    dict(type='Collect', keys=['img{}'.format(i) for i in range(18)] + \
                            ['depth_map{}'.format(i) for i in range(18)] + \
                            ['sf_map{}'.format(i) for i in range(18)] + \
                            ['cam_intrinsic', 'cam_pose']),
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuscSpatialTempV2',
        sf_path='/public/MARS/datasets/nuScenes-SF/trainval',
        img_path='data/nuscenes/',
        pose_path='/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic_v2.json',
        pipeline=train_pipeline,
        training=True,
    ),
    val=dict(
        type='NuscSpatialTempV2',
        sf_path='/public/MARS/datasets/nuScenes-SF/trainval',
        img_path='data/nuscenes/',
        pose_path='/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic_v2.json',
        pipeline=val_pipeline,
        training=False,
    ),
    test=dict(
        type='NuscSpatialTempV2',
        sf_path='/public/MARS/datasets/nuScenes-SF/trainval',
        img_path='data/nuscenes/',
        pose_path='/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic_v2.json',
        pipeline=val_pipeline,
        training=False,
        #samples_per_gpu=16,
    ),
)

checkpoint_config = dict(interval=2)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook2')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
#workflow = [('train', 1), ('val', 1)]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.001)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[24, 32],
)
momentum_config = None

# runtime settings
total_epochs = 40
#load_from='/public/MARS/surrdet/tyz/depth-net.pth'
load_from=None
