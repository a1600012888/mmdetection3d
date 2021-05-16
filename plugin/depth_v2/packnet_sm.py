import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .utils import get_depth_metrics

class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))


class ResidualConv(nn.Module):
    """2D Convolutional residual block with GroupNorm and ELU"""
    def __init__(self, in_channels, out_channels, stride, dropout=None):
        """
        Initializes a ResidualConv object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride
        dropout : float
            Dropout value
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        """Runs the ResidualConv layer."""
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return self.activ(self.normalize(x_out + shortcut))


def ResidualBlock(in_channels, out_channels, num_blocks, stride, dropout=None):
    """
    Returns a ResidualBlock with various ResidualConv layers.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_blocks : int
        Number of residual blocks
    stride : int
        Stride
    dropout : float
        Dropout value
    """
    layers = [ResidualConv(in_channels, out_channels, stride, dropout=dropout)]
    for i in range(1, num_blocks):
        layers.append(ResidualConv(out_channels, out_channels, 1, dropout=dropout))
    return nn.Sequential(*layers)


class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        return self.activ(x) / self.min_depth

########################################################################################################################

def packing(x, r=2):
    """
    Takes a [B,C,H,W] tensor and returns a [B,(r^2)C,H/r,W/r] tensor, by concatenating
    neighbor spatial pixels as extra channels. It is the inverse of nn.PixelShuffle
    (if you apply both sequentially you should get the same tensor)
    Parameters
    ----------
    x : torch.Tensor [B,C,H,W]
        Input tensor
    r : int
        Packing ratio
    Returns
    -------
    out : torch.Tensor [B,(r^2)C,H/r,W/r]
        Packed tensor
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

########################################################################################################################

class PackLayerConv2d(nn.Module):
    """
    Packing layer with 2d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2):
        """
        Initializes a PackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels * (r ** 2), in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)

    def forward(self, x):
        """Runs the PackLayerConv2d layer."""
        x = self.pack(x)
        x = self.conv(x)
        return x


class UnpackLayerConv2d(nn.Module):
    """
    Unpacking layer with 2d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2):
        """
        Initializes a UnpackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * (r ** 2), kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        """Runs the UnpackLayerConv2d layer."""
        x = self.conv(x)
        x = self.unpack(x)
        return x

########################################################################################################################

class PackLayerConv3d(nn.Module):
    """
    Packing layer with 3d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8):
        """
        Initializes a PackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the PackLayerConv3d layer."""
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x


class UnpackLayerConv3d(nn.Module):
    """
    Unpacking layer with 3d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        """
        Initializes a UnpackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the UnpackLayerConv3d layer."""
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x


@DETECTORS.register_module()
class PackNetSlim02(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).
    Slimmer version, with fewer feature channels
    https://arxiv.org/abs/1905.02693
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, **kwargs):
        super().__init__()
        self.version = version
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 32, out_channels
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)
        # Support for different versions
        if self.version == 'A':  # Channel concatenation
            n1o, n1i = n1, n1 + ni + no
            n2o, n2i = n2, n2 + n1 + no
            n3o, n3i = n3, n3 + n2 + no
            n4o, n4i = n4, n4 + n3
            n5o, n5i = n5, n5 + n4
        elif self.version == 'B':  # Channel addition
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

        # Decoder

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_pred(self, x):
        x = self.pre_calc(x)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        #print(disp1.shape)
        return [disp1, disp2, disp3, disp4]

    #计算值的inverse
    def inv_val(self, x):
        y = []
        for v in x:
            y.append(1. / v.clamp(min=1e-6))   #避免除数为0
        return y

    def get_L1_loss(self, preds, label, mask):
        loss = 0
        num = torch.sum(mask)
        num = num * 1.0
        #print(label.shape)
        B, _, H, W = label.shape
        if isinstance(preds, list):     #train
            for pred in preds:
                #print(pred.shape)
                up_pred = nn.functional.interpolate(pred, size=[H, W])
                loss1 = torch.abs((label - up_pred)) * mask
                loss1 = torch.sum(loss1) / num
                loss += loss1
        else:                           #test
            loss = torch.abs((label - preds)) * mask
            loss = torch.sum(loss) / num

        return loss

    def _gradient_x(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def _gradient_y(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

#edge-aware smoothness loss
#reference code in https://github.com/google-research/google-research/blob/f4318b248d5e588b6c4ce1f3de0838b5db6139f6/depth_and_motion_learning/losses/regularizers.py#L27
    def get_smooth_loss(self, preds, img):
        loss = 0
        B, _, H, W = img.shape
        weights_x = torch.exp(-abs(self._gradient_x(img)))
        weights_y = torch.exp(-abs(self._gradient_y(img)))
        if isinstance(preds, list):
            for pred in preds:
                up_pred = nn.functional.interpolate(pred, size=[H, W])
                dep_dx = abs(self._gradient_x(up_pred))
                dep_dy = abs(self._gradient_y(up_pred))
                loss1 = torch.sum(dep_dx * weights_x) / torch.numel(dep_dx)
                loss1 += torch.sum(dep_dy * weights_y) / torch.numel(dep_dy)
                loss += loss1
        else:
            dep_dx = abs(self._gradient_x(preds))
            dep_dy = abs(self._gradient_y(preds))
            loss = torch.sum(dep_dx * weights_x) / torch.numel(dep_dx)
            loss += torch.sum(dep_dy * weights_y) / torch.numel(dep_dy)
        return loss


    def forward(self, return_loss=True, rescale=False, **kwargs):
        
        if not return_loss:
            # in evalhook!

            x = kwargs['img']
            label = kwargs['depth_map']

            data = {'img':x, 'depth_map':label}
            inv_depth_pred = self.get_pred(data['img'])
            depth_pred = self.inv_val(inv_depth_pred)
            label = data['depth_map'].unsqueeze(dim=1)
            mask = (label > 0)

            smoothness_loss = self.get_smooth_loss(depth_pred, data['img'])
            L1_loss = self.get_L1_loss(depth_pred, label, mask)
            loss = L1_loss + smoothness_loss
            #print('L1_loss', depth_pred[0].shape, label.shape, mask.shape)
            with torch.no_grad():
                metrics = get_depth_metrics(depth_pred[0], label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]

            # hack the hook
            # outputs[0]=None. see https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/test.py#L99
            #outputs = {'loss': loss, 'log_vars':log_vars, 'num_samples':depth_pred.size(0), 0:None}
            #print('val', loss)
            metrics.append(loss.item())
            return [metrics]
        raise NotImplementedError


    def train_step(self, data, optimzier):
        inv_depth_pred = self.get_pred(data['img'])
        depth_pred = self.inv_val(inv_depth_pred)
        label = data['depth_map'].unsqueeze(dim=1)
        mask = (label > 0)

        smoothness_loss = self.get_smooth_loss(depth_pred, data['img'])
        L1_loss = self.get_L1_loss(depth_pred, label, mask)
        loss = L1_loss + smoothness_loss
        #print('train_loss', depth_pred[0].shape)
        with torch.no_grad():
                metrics = get_depth_metrics(depth_pred[0], label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]
                abs_diff, abs_rel, sq_rel, rmse, rmse_log = metrics
        
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)

        std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
        mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
        img = data['img'] * std + mean
        img = img / 255.0
        depth_at_gt = depth_pred[0] * mask

        log_vars = {'loss': loss.item(), 'L1_loss': L1_loss.item(), 
                    'sm_loss': smoothness_loss.item(), 'sparsity': sparsity.item(),
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log
                     }
        
        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        outputs = {'pred': torch.clamp(1.0 / (depth_pred[0]+1e-4), 0, 1), 'data': img,
                'label': torch.clamp(1.0 / (label+1e-4), 0, 1),
                'depth_at_gt': torch.clamp(1.0 / (depth_at_gt+1e-4), 0., 1),
                'loss':loss, 'log_vars':log_vars, 'num_samples':depth_pred[0].size(0)}

        return outputs

    def val_step(self, data, optimizer):

        inv_depth_pred = self.get_pred(data['img'])
        depth_pred = self.inv_val(inv_depth_pred)
        label = data['depth_map'].unsqueeze(dim=1)
        mask = (label > 0)

        #print(depth_pred.shape, label.shape, mask.shape, 'data shape')
        #from IPython import embed
        #embed()
        #loss = torch.abs((label - depth_pred)) * mask
        #loss = torch.sum(loss) / torch.sum(mask)
        loss = self.get_L1_loss(depth_pred[0], label, mask)
        #print('L1_loss', depth_pred[0].shape, label.shape, mask.shape)
        
        with torch.no_grad():
                metrics = get_depth_metrics(depth_pred[0], label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]
                abs_diff, abs_rel, sq_rel, rmse, rmse_log = metrics
        
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)

        std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
        mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
        img = data['img'] * std + mean
        img = torch.clamp(img, min=0.0, max=255.0)
        img = img / 255.0
        depth_at_gt = depth_pred[0] * mask

        log_vars = {'loss': loss.item(), 'sparsity': sparsity.item(),
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log
                     }
        
        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        outputs = {'pred': torch.clamp(1.0 / (depth_pred[0]+1e-4), 0, 1), 'data': img,
                'label': torch.clamp(1.0 / (label+1e-4), 0, 1),
                'depth_at_gt': torch.clamp(1.0 / (depth_at_gt+1e-4), 0., 1),
                'loss':loss, 'log_vars':log_vars, 'num_samples':depth_pred[0].size(0)}

        return outputs