import torch
from torch._C import NoneType
from torchvision import utils as vutils
import numpy as np

import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import time

def convert2RGB(img):
    rgb = torch.zeros((40, 3))
    i = 0
    for r in [0, 122, 255]:
        for g in [0, 122, 255]:
            for b in [0, 85, 170, 255]:
                rgb[i][0] = r
                rgb[i][1] = g
                rgb[i][2] = b
                i = i + 1
    #print(rgb.size(), rgb[1].size())
    B, C, H, W = img.size()
    img = img.permute(0, 2, 3, 1).reshape(B*H*W, C)
    #print(img.size())
    cls_index = img[:, 0].to(torch.long)
    img = rgb[cls_index]
    img = img / 255
    img = img.view(B, H, W, C).permute(0, 3, 1, 2)
    return img



def visulize(pred, img_metas):
    
    img_root = 'saved_imgs/'
    
    #gt_seg = gt_seg.repeat(1, 3, 1, 1)
    #print('gt_seg', gt_seg.size())
    #gt_seg = convert2RGB(gt_seg)
    
    filename = img_metas[0]['filename'].split('/')
    pred_value, pred_label = pred.topk(1, dim=1)
    # shape (B, 3, H, W)
    pred_label = pred_label.repeat(1, 3, 1, 1)
    print('pred_label', pred_label.size())
    pred_seg = convert2RGB(pred_label)
    print(filename)
    vutils.save_image(pred_seg, img_root+'pred_'+filename[-1])
    #vutils.save_image(gt_seg, img_root+'label_'+filename[-1])


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)



def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=False,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img_name = img.split('/')[-1]
    #print(img)
    img = mmcv.imread(img).astype(np.uint8)
    #img = img.reshape(1600, 900, 3)
    #print('img', img.shape)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    #fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            for obj in segms[labels[i]]:
                mask = obj.astype(bool)
                #print('mask', mask.shape)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    vutils.save_image(img, 'save_imgs/'+'test_'+img_name)
    return img

def show_mask(img, segms, num_classes=10, out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        show (bool): Whether to show the image. Default: True
        out_file (str, optional): The filename to write the image.
            Default: None
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    img_name = img.split('/')[-1]
    img = mmcv.imread(img).astype(np.uint8)
    #img = img.reshape(1600, 900, 3)
    #print('img', img.shape)

    mask_colors = []
    # Get random state before set seed, and restore random state later.
     # Prevent loss of randomness.
    # See: https://github.com/open-mmlab/mmdetection/issues/5844
    state = np.random.get_state()
    # random color
    np.random.seed(42)
    mask_colors = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_classes)
    ]
    np.random.set_state(state)


    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    for i in range(num_classes):
        if segms is not None:
            color_mask = mask_colors[i]
            for obj in segms[i]:
                mask = obj.astype(bool)
                #print('mask', mask.shape)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

    img = mmcv.rgb2bgr(img)
    img = torch.from_numpy(img)
    #if out_file is not None:
    #    mmcv.imwrite(img, out_file)
    img = img / 255
    img = img.unsqueeze(dim=0).permute(0, 3, 1 ,2)
    vutils.save_image(img, 'saved_imgs/'+'nus_'+img_name)
    print(img_name)
    return img

def get_mask_label(img_metas, segms, num_classes=10):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
    Returns:
        Tensor: one-hot label [H, W, num_classes]
    """
    width, height = img_metas['ori_shape'][1], img_metas['ori_shape'][0]

    img_mask_label = torch.zeros(height, width, num_classes)
    one_hot_label = torch.eye(10)
    for i in range(num_classes):
        for obj in segms[i]:
            mask = obj.astype(bool)
            img_mask_label[mask] = one_hot_label[i]

    return img_mask_label

def get_painted_pts_bev(points, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    mask = ((points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3]) & 
        (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] > pc_range[2]) & (points[:, 2] < pc_range[5]))
    pts = points[mask]
    points_2d = pts[:, :2]
    #print(points[i].size(), pts.size(), points_2d.size())
    points_2d[:, 0] = points_2d[:, 0] - pc_range[0]
    points_2d[:, 1] = points_2d[:, 1] - pc_range[1]

    res = 0.05 
    x_img = (points_2d[:, 0] / res).long()
    y_img = (points_2d[:, 1] / res).long()
 
    # 创建图像数组
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(x_max, y_max, 3)
    im[x_img, y_img, :] = 1

    rgb = torch.zeros(10, 3)
    rgb[0] = torch.Tensor([1, 0, 0])
    rgb[1] = torch.Tensor([0, 1, 0])
    rgb[2] = torch.Tensor([0, 0, 1])
    rgb[3] = torch.Tensor([0, 1, 1])
    rgb[4] = torch.Tensor([1, 0, 1])
    rgb[5] = torch.Tensor([1, 1, 0])
    rgb[6] = torch.Tensor([0, 0.5, 1])
    rgb[7] = torch.Tensor([0.5, 0, 1])
    rgb[8] = torch.Tensor([0.5, 0.5, 1])
    rgb[9] = torch.Tensor([0.5, 1, 0.5])

    #print(one_hot)
    #for i in range(10):
    #    print(pts[i])
    #print('pts', pts.size())
    for i in range(10):
        color_mask = (pts[:, 5+i] == 1)
        print('color_mask', color_mask.sum())
        painted_pts = points_2d[color_mask]
        x_img = (painted_pts[..., 0:1] / res).long()
        y_img = (painted_pts[..., 1:2] / res).long()
        for j in range(-1, 2):
            for k in range(-1, 2):
                im[x_img+j, y_img+k] = rgb[i]
        #im[x_img, y_img] = rgb[i]
    # shape [H, W, 3] --> [3, H, W]
    im = im.permute(2, 0 ,1)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print(timestamp)
    vutils.save_image(im, 'painted_sweep_bev/' + timestamp + '.jpg')
    return im