import torch
import torch.nn as nn
import torch.nn.functional as F

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class ResBlockGnorm(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlockGnorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(16, planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(16, planes)
        self.prelu2 = nn.PReLU(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(16, self.expansion * planes)
            )

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.prelu2(out)
        return out

class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResBlockGnorm(c, c), 
            ResBlockGnorm(c, c), 
            ResBlockGnorm(c, c), 
            ResBlockGnorm(c, c), 
        )
        self.conv1 = nn.ConvTranspose2d(c, c, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return flow


class SFNet(nn.Module):

    def __init__(self, **kwargs):
        super(SFNet, self).__init__()

        self.conv1 = conv(8, 16)
        self.conv_block1 = IFBlock(in_planes=16, c=192, scale=4)
        self.conv_block2 = IFBlock(in_planes=192, c=48, scale=2)
        self.conv_block3 = IFBlock(in_planes=48, c=16, scale=1)
        self.final_head = nn.Conv2d(16, 3, 3, 1, 1)

    def get_pred(self, img): #img1, img2):
        '''
        img: [N, 12, H, W] cat[now_img_depth, rec_img_depth, next_img_depth]
        '''
        #img = torch.cat([img1, img2], dim=1)
        x = self.conv1(img)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        pred = self.final_head(x)

        return pred

    def forward(self, img):
        return self.get_pred(img)
