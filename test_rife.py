import torch
import numpy as np
import cv2
from models.IFNet_HDv3 import IFNet
from models.FastFlowNet.models.FastFlowNet_v2 import FastFlowNet

torch.set_grad_enabled(False)

ifnet = IFNet().cuda()
ifnet.load_state_dict(torch.load(r'weights\rife.pkl'))
flownet = FastFlowNet().cuda().eval()
flownet.load_state_dict(torch.load(r'weights\fastflownet_ft_sintel.pth'))

hw = (576, 960)  # 计算光流传入的图片大小，fastflownet要求必须被64整除
iwh = (1920, 1088)  # 传入RIFE补帧网络的图片大小(cv2)
owh = (1920, 1080)  # 最后输出的图片大小(cv2)

# 光流计算
def calc_flow(model, a, b):
    def centralize(img1, img2):
        b, c, h, w = img1.shape
        rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    a, b, _ = centralize(a, b)

    input_t = torch.cat([a, b], 1)

    output = model(input_t).data

    flow = 20.0 * output

    return flow

# 光流距离计算
def distance_calculator(x):
    u, v = x[:, 0:1], x[:, 1:]
    return torch.sqrt(u ** 2 + v ** 2)

img0, img1, img2 = map(cv2.imread, ['input/01.png', 'input/02.png', 'input/03.png'])

# RIFE用
img0, img1, img2 = map(cv2.resize, [img0, img1, img2], [iwh] * 3)
I0, I1, I2 = map(lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.,
                     [img0, img1, img2])

# 计算光流用
I0f, I1f, I2f = map(lambda x: torch.nn.functional.interpolate(x, size=hw, mode='bilinear', align_corners=False),
                         [I0, I1, I2])

flow10 = calc_flow(flownet, I1f, I0f)
flow12 = calc_flow(flownet, I1f, I2f)

# 计算 i1相对i0,i2的时刻t
d10 = distance_calculator(flow10)
d12 = distance_calculator(flow12)
t = d10 / (d10 + d12)
t = torch.nn.functional.interpolate(t, size=I0.shape[2:], mode='bilinear', align_corners=False)

# 计算i0,i1的中间帧i01, ...
I01 = ifnet(torch.cat((I0, I1), 1), t, scale_list=[8, 4, 2, 1])
I12 = ifnet(torch.cat((I1, I2), 1), t, scale_list=[8, 4, 2, 1])

# 输出连续五帧
cnt = 0
for item in [I0, I01, I1, I12, I2]:
    cv2.imwrite(f'output/{cnt:02d}.png',
                cv2.resize((item[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), owh))
    cnt += 1