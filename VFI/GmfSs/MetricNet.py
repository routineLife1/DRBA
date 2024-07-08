import torch
import torch.nn as nn
import torch.nn.functional as F

backwarp_cache = {}

def backwarp(img, flow):
    k = (str(img.device), str(flow.shape))
    B, _, H, W = flow.shape
    if k in backwarp_cache:
        grid = backwarp_cache[k]
    else:
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        backwarp_cache[k] = grid
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10, mode='net'):
        metric0 = torch.nn.functional.l1_loss(img0, backwarp(img1, flow01), reduction='none').mean([1], True)
        metric1 = torch.nn.functional.l1_loss(img1, backwarp(img0, flow10), reduction='none').mean([1], True)
        if mode == 'net':
            metric0 = self.metric_net(torch.cat((img0, -metric0), 1))
            metric1 = self.metric_net(torch.cat((img1, -metric1), 1))
        elif mode == 'alpha':
            metric0 = -20 * metric0
            metric1 = -20 * metric1
        return metric0, metric1
