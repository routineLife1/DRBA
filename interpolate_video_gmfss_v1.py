import os
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import torch
import numpy as np
import time
import warnings
from models.model_nb222.softsplat import softsplat as warp

warnings.filterwarnings("ignore")
from models.model_nb222.RIFE import Model

input = r'E:\[Up to 21°C] 敗北女角太多了！ - 03 (Baha 1920x1080 AVC AAC MP4) [F933FE40].mp4'
save = r'D:\tmp\output'
scale = 1.0
times = 5  # 补帧倍数(必须是整数倍)
global_size = (960, 540)  # 全局图像尺寸
# swap_thresh means Threshold for applying the swap mask.
# 0 means fully apply the swap mask.
# 0.n means enable swapping when the timestep difference is greater than 0.n.
# 1 means never apply the swap mask.
swap_thresh = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model = Model()
model.load_model(r'weights\train_log_nb222', -1)
model.device()
model.eval()


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.


# 加载图像
def load_image(img, _scale):
    h, w, _ = img.shape
    while h * _scale % 64 != 0:
        h += 1
    while w * _scale % 64 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    img = to_tensor(img)
    return img


output_counter = 0  # 输出计数器


def put(things):  # 将输出帧推送至write_buffer
    global output_counter
    output_counter += 1
    write_buffer.put([output_counter, things])


def get():  # 获取输入帧
    return read_buffer.get()


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    while True:
        item = w_buffer.get()
        if item is None:
            break
        num = item[0]
        content = item[1]
        cv2.imwrite(os.path.join(save, "{:0>9d}.png".format(num)), cv2.resize(content, global_size))


@torch.autocast(device_type="cuda")
def make_inference(_I0, _I1, _I2, _scale):
    # Flow distance calculator
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    reuse_i1i0 = model.reuse(_I1, _I0, scale)
    reuse_i1i2 = model.reuse(_I1, _I2, scale)

    flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
    flow12, metric12 = reuse_i1i2[0], reuse_i1i0[2]

    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)

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
    _output = list()
    if times % 2:
        for i in range((times - 1) // 2):
            t = (i + 1) / times
            # Adjust timestep parameters for interpolation between frames I0, I1, and I2
            # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2
            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                              timestep1=1 - (t * 2) * drm01, swap_thresh=swap_thresh))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                              timestep1=1 - (t * 2) * drm21, swap_thresh=swap_thresh))
            _output = list(reversed(output1)) + [_I1] + output2
    else:
        for i in range(times // 2):
            t = (i + 0.5) / times
            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                              timestep1=1 - (t * 2) * drm01, swap_thresh=swap_thresh))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                              timestep1=1 - (t * 2) * drm21, swap_thresh=swap_thresh))
            _output = list(reversed(output1)) + output2

    _output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), _output)

    return _output


video_capture = cv2.VideoCapture(input)
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

cnt = 1
# start inference
i0, i1 = get(), get()
I0, I1 = load_image(i0, scale), load_image(i1, scale)

# head
output = make_inference(I0, I1, I1, scale)
for x in output:
    put(x)
pbar.update(1)

while True:
    i2 = get()
    if i2 is None:
        break
    I2 = load_image(i2, scale)

    output = make_inference(I0, I1, I2, scale)
    for x in output:
        put(x)

    i0, i1 = i1, i2
    I0, I1 = I1, I2
    pbar.update(1)

# tail(结束时i0,i1已经移动到-2,-1帧的位置)
output = make_inference(I0, I1, I1, scale)
for x in output:
    put(x)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
