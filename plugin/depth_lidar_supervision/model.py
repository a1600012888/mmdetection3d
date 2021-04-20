from mmdet.models import DETECTORS
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16

@DETECTORS.register_module()
class ResUNet(nn.Module):

    def __init__(self, ):
        super(ResUNet, self).__init__()
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs)

    def train_step(self, data, optimizer):
        outputs = {loss: 0, log_vars={'loss_name': 0}, num_samples=len(data['imgs'])}
        return outputs

    def val_step(self, data, optimizer):