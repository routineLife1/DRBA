import torch
import numpy as np
import cv2
from models.model_nb222.RIFE import Model

torch.set_grad_enabled(False)

model = Model()
model.load_model(r'weights\train_log_nb222', -1)
model.device()
model.eval()

hw = (576, 960)  # 计算光流传入的图片大小，fastflownet要求必须被64整除
iwh = (1920, 1088)  # 传入RIFE补帧网络的图片大小(cv2)
owh = (1920, 1080)  # 最后输出的图片大小(cv2)
scale = 1.0  # 光流尺度

# 光流距离计算
def distance_calculator(x):
    u, v = x[:, 0:1], x[:, 1:]
    return torch.sqrt(u ** 2 + v ** 2)

img0, img1, img2 = map(cv2.imread, ['input/01.png', 'input/02.png', 'input/03.png'])

# RIFE用
img0, img1, img2 = map(cv2.resize, [img0, img1, img2], [iwh] * 3)
I0, I1, I2 = map(lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.,
                     [img0, img1, img2])


reuse_i0i1 = model.reuse(I0, I1, scale)
reuse_i1i2 = model.reuse(I1, I2, scale)

flow10 = reuse_i0i1[1]
flow12 = reuse_i1i2[0]

# 计算 i1相对i0,i2的时刻t
d10 = distance_calculator(flow10)
d12 = distance_calculator(flow12)
t = d10 / (d10 + d12)
# t = torch.nn.functional.interpolate(t, size=I0.shape[2:], mode='bilinear', align_corners=False)

# 计算i0,i1的中间帧i01, ...
I01 = model.inference(I0, I1, reuse_i0i1, t)
I12 = model.inference(I1, I2, reuse_i1i2, t)

# 输出连续五帧
cnt = 0
for item in [I0, I01, I1, I12, I2]:
    cv2.imwrite(f'output/{cnt:02d}.png',
                cv2.resize((item[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), owh))
    cnt += 1