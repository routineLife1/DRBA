import torch
import numpy as np
import cv2
from models.model_nb222.RIFE import Model
from models.model_nb222.softsplat import softsplat as warp

torch.set_grad_enabled(False)

model = Model()
model.load_model(r'weights\train_log_nb222', -1)
model.device()
model.eval()

iwh = (1920, 1088)  # input resolution
owh = (1920, 1080)  # output resolution
scale = 1.0  # flow scale
n = 3  # times
# swap_thresh means Threshold for applying the swap mask.
# 0 means fully apply the swap mask.
# 0.n means enable swapping when the timestep difference is greater than 0.n.
# 1 means never apply the swap mask.
swap_thresh = 0.5


# calc distance by flow
def distance_calculator(x):
    u, v = x[:, 0:1], x[:, 1:]
    return torch.sqrt(u ** 2 + v ** 2)


img0, img1, img2 = map(cv2.imread, ['input/01.png', 'input/02.png', 'input/03.png'])

img0, img1, img2 = map(cv2.resize, [img0, img1, img2], [iwh] * 3)
I0, I1, I2 = map(lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.,
                 [img0, img1, img2])

reuse_i1i0 = model.reuse(I1, I0, scale)
reuse_i1i2 = model.reuse(I1, I2, scale)

flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
flow12, metric12 = reuse_i1i2[0], reuse_i1i0[2]

# Compute the distance using the optical flow and distance calculator
d10 = distance_calculator(flow10) + 1e-4
d12 = distance_calculator(flow12) + 1e-4

# Calculate the distance ratio map
drm10 = d10 / (d10 + d12)
drm12 = d12 / (d10 + d12)

# The distance ratio map (drm) is initially aligned with I1.
# To align it with I0 and I2, we need to warp the drm maps.
# Note: To reverse the direction of the drm map, use 1 - drm and then warp it.
drm01 = warp(1 - drm10, flow10, metric10, strMode='soft')
drm21 = warp(1 - drm12, flow12, metric12, strMode='soft')

# Create a mask with all ones to identify the holes in the warped drm maps
ones_mask = torch.ones_like(drm01, device=drm01.device)

# Warp the ones mask
warped_ones_mask01 = warp(ones_mask, flow10, metric10, strMode='soft')
warped_ones_mask21 = warp(ones_mask, flow12, metric12, strMode='soft')

# Identify holes in warped drm map
holes01 = warped_ones_mask01 < 0.999
holes21 = warped_ones_mask21 < 0.999

# Fill the holes in the warped drm maps with the inverse of the original drm maps
drm01[holes01] = (1 - drm10)[holes01]
drm21[holes21] = (1 - drm12)[holes21]

output1, output2 = list(), list()
output = list()
if n % 2:
    for i in range((n - 1) // 2):
        t = (i + 1) / n
        # Adjust timestep parameters for interpolation between frames I0, I1, and I2
        # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2
        output1.append(model.inference_t2(I1, I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                          timestep1=1 - (t * 2) * drm01, swap_thresh=swap_thresh))
        output2.append(model.inference_t2(I1, I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                          timestep1=1 - (t * 2) * drm21, swap_thresh=swap_thresh))
        output = list(reversed(output1)) + [I1] + output2
else:
    for i in range(n // 2):
        t = (i + 0.5) / n
        output1.append(model.inference_t2(I1, I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                          timestep1=1 - (t * 2) * drm01, swap_thresh=swap_thresh))
        output2.append(model.inference_t2(I1, I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                          timestep1=1 - (t * 2) * drm21, swap_thresh=swap_thresh))
        output = list(reversed(output1)) + output2

cnt = 0
for out in output:
    cv2.imwrite(f'output/{cnt:03d}.png',
                cv2.resize((out[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), owh))
    cnt += 1
