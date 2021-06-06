import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

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

class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False)
        return flow
    

class SFNet(nn.Module):
    def __init__(self):
        super(SFNet, self).__init__()
        self.block0 = IFBlock(8, scale=4, c=320)
        self.block1 = IFBlock(10, scale=2, c=225)
        self.block2 = IFBlock(10, scale=1, c=135)

    def forward(self, x):
        '''
        x: [N, 8, H, W]
        '''
        img_0, img_1 = x[:, :4], x[:, 4:]

        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        
        warped_img1 = warp(img_1, F1_large)
        flow1 = self.block1(torch.cat((img_0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        
        warped_img1 = warp(img_1, F2_large)
        flow2 = self.block2(torch.cat((img_0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)
        
        ret_flows = [F.interpolate(ret_flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0 for ret_flow in [F3, F2, F1]]
        
        return ret_flows
    
    def get_optical_pred(self, x):
        return self(x)
    
    def upgrade_flow(self, x, optical_flows):
        disp_0, disp_1 = x[:, [3]], x[:, [7]]
        depth_0 = 1.0 / torch.clamp(disp_0, min=1e-5)
        depth_1 = 1.0 / torch.clamp(disp_1, min=1e-5)

        ret = []
        for op_flow in optical_flows:
            warped_depth1 = warp(depth_1, op_flow)
            flow_z = warped_depth1 - depth_0
            ret.append(torch.cat([op_flow, flow_z], dim=1))
        
        return ret

