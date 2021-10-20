import os

import imageio


def make_sample_videos(sample_tokens: list,
                       nusc_explorer,
                       out_dir='../../work_dirs/visualization/test',
                       fps=2):
    out_img_dir = os.path.join(out_dir, 'images')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    gif_path = os.path.join(out_dir, 'final.gif')
    
    out_img_paths = []
    for i, sample_token in enumerate(sample_tokens):
        out_path = os.path.join(out_img_dir, '{}.png'.format(i))
        out_img_paths.append(out_path)
        nusc_explorer.render_sample(sample_token, out_path=out_path)
    
    img_cat_list = []
    for img_path in out_img_paths:
        img_cat_list.append(imageio.imread(img_path))
    
    imageio.mimsave(gif_path, img_cat_list, fps=fps)


def make_sensor_videos(sample_tokens: list, 
                       nusc_explorer,
                       out_dir='../../work_dirs/visualization/test',
                       fps=2):
    
    out_img_dir = os.path.join(out_dir, 'images')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    gif_path = os.path.join(out_dir, 'final.gif')
    
    out_img_paths = []
    for i, sample_token in enumerate(sample_tokens):
        out_path = os.path.join(out_img_dir, '{}.png'.format(i))
        out_img_paths.append(out_path)
        nusc_explorer.render_sample_data(sample_token, out_path=out_path)
    
    img_cat_list = []
    for img_path in out_img_paths:
        img_cat_list.append(imageio.imread(img_path))
    
    imageio.mimsave(gif_path, img_cat_list, fps=fps)
    

def make_sensor_pred_videos(sample_tokens: list, 
                            results_keys: list,
                            results_dict,
                            nusc_explorer,
                            out_dir='../../work_dirs/visualization/test',
                            fps=2):
    
    out_img_dir = os.path.join(out_dir, 'images')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    gif_path = os.path.join(out_dir, 'final.gif')
    
    out_img_paths = []
    for i, (sample_token, key) in enumerate(zip(sample_tokens, results_dict)):
        out_path = os.path.join(out_img_dir, '{}.png'.format(i))
        out_img_paths.append(out_path)
        results = results_dict[key]
        nusc_explorer.render_sample_pred(
            sample_token, results, out_path=out_path)
    
    img_cat_list = []
    for img_path in out_img_paths:
        img_cat_list.append(imageio.imread(img_path))
    
    imageio.mimsave(gif_path, img_cat_list, fps=fps)
    
def _test():
    from tools.visualization.nusc_explorer import NuScenesMars, NuScenesExplorerMars

    nusc = NuScenesMars(
        version='v1.0-trainval', dataroot='../../data/nuscenes')
    nusc_exp = NuScenesExplorerMars(nusc)
    
    samples = nusc.sample
    samples[0]
    
    sample_token_list = [a['token'] for a in samples[:20]]
    sample_cam_token_list = [a['data']['CAM_FRONT'] for a in samples[:20]]
    make_sensor_videos(sample_cam_token_list, nusc_exp, out_dir='../../work_dirs/visualization/test_cam')
    make_sample_videos(sample_token_list, nusc_exp, out_dir='../../work_dirs/visualization/test')