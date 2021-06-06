import argparse
from matplotlib.pyplot import sca
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from tools.surrdet.utils import disp2rgb
from PIL import Image
import numpy as np
from mmcv.parallel.scatter_gather import scatter
import json
from tqdm import tqdm

CamNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 
  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT']


def parse_output(output, save_dir):
    """
    in model.eval_forward:
    outputs = {'inv_depth_preds': now_inv_depth_preds, 
                    'cam_intrinsic': now_cam_intrin, 
                    'cam_pose': now_cam_pose, 
                    'imgs': now_imgs}
    """
    
    disp_preds = output['inv_depth_preds'][0] # [6*1, 1, H, W]
    imgs = output['imgs'].permute(0, 2, 3, 1).detach().cpu().numpy() # [6*1, H, W, C]

    disp_rgb = disp2rgb(disp_preds) # [6*1, H, W, C]
    
    #from IPython import embed
    #embed()

    disp_rgb_list = np.split(disp_rgb, 6, axis=0)
    disp_rgb = np.concatenate(disp_rgb_list, axis=2)[0]

    img_list = np.split(imgs, 6, axis=0) # [1, H, W, 3]
    imgs = np.concatenate(img_list, axis=2)[0]

    visual_map = np.concatenate([disp_rgb, imgs], axis=0) * 255.0
    visual_map = visual_map.astype(np.uint8)

    visual_im = Image.fromarray(visual_map)

    visual_im.save(os.path.join(save_dir, 'visualize.png'))

    return disp_rgb_list, img_list, visual_im


def merge_output(merged_output_list, save_dir):

    list_of_img_seq = [[], [], [], [], [], []]
    for output_list in merged_output_list:
        disp_rgb_list, img_list, visual_im = output_list
        i = 0
        for disp, img in zip(disp_rgb_list, img_list):
            ret = np.concatenate([disp, img], axis=1)[0] * 255 # [1, 2H, W, C]
            im = Image.fromarray(ret.astype(np.uint8))
            list_of_img_seq[i].append(im)
            i += 1
        
    for img_seq, cam_name in zip(list_of_img_seq, CamNames):
        
        img_seq[0].save(os.path.join(save_dir, '{}.gif'.format(cam_name)), format='GIF', append_images=img_seq[1:], save_all=True)
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out_dir', help='output parse files', default='/public/MARS/datasets/nuScenes-SF/SF-Input')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin') & cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

        
    # import modules from string list.
    # if cfg.get('custom_imports', None):
    #     from mmcv.utils import import_modules_from_strings
    #     import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    #from IPython import embed
    #embed()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    #if args.checkpoint is not None:
    #    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    
    
    model.eval()
    meta_json = {}

    print('len of data loader: ', len(data_loader))
    for i, data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            data = scatter(data, [-1])[0]
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            
            key_img_path = data['img_metas'][0]['filename']
            key_img_name = os.path.join(*key_img_path.split('/')[2:])
            key_img_filename = key_img_path.split('/')[-1]

            save_path = os.path.join(args.out_dir, key_img_filename)
            outputs = model.module.preprocess_forward(data)
            outputs = outputs.detach().cpu().numpy()
            np.save(save_path, outputs)
            meta_json[key_img_name] = save_path + '.npy'
    
    with open(os.path.join(args.out_dir, 'sf_inp_val_meta_{}.json'.format(args.local_rank)), 'w') as f:
        json.dump(meta_json, f)
    

if __name__ == '__main__':
    main()
