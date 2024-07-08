# for real-time playback(+TensorRT)

import os
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import torch
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")
from models.IFNet_HDv3 import IFNet
from models.FastFlowNet.models.FastFlowNet_v2 import FastFlowNet

input = r'E:\[Up to 21°C] 擅長逃跑的殿下 - 01 (Baha 1920x1080 AVC AAC MP4) [8236997B].mp4'
save = r'D:\tmp\output'
scale = 1.0
exp = 3  # 补帧为2的exp次方倍率
global_size = (960, 540)  # 全局图像尺寸
scene_detection = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

ifnet = IFNet().cuda().eval()
ifnet.load_state_dict(torch.load(r'weights\rife.pkl'))
flownet = FastFlowNet().cuda().eval()
flownet.load_state_dict(torch.load(r'weights\fastflownet_ft_sintel.pth'))


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
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    def get_timesteps(seq, t, n):
        for j in range(n):
            iseq = [seq[i] + (seq[i + 1] - seq[i]) * t for i in range(0, len(seq) - 1)]
            k = 0
            while len(iseq):
                seq.insert(2 * k + 1, iseq.pop(0))
                k += 1

        seq = [torch.round(s, decimals=4) for s in seq]

        return seq[1:-1]

    # 576, 960尺寸计算就行，速度很快
    I0f, I1f, I2f = map(
        lambda x: torch.nn.functional.interpolate(x, size=(576, 960), mode='bilinear', align_corners=False),
        [_I0, _I1, _I2])

    flow10 = calc_flow(flownet, I1f, I0f)
    flow12 = calc_flow(flownet, I1f, I2f)

    # 计算 i1相对i0,i2的时刻t
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)
    t = d10 / (d10 + d12)
    t = torch.nn.functional.interpolate(t, size=_I0.shape[2:], mode='bilinear', align_corners=False)

    t0, t1 = t.clone(), t.clone()

    if scene_detection:
        left_scene = torch.abs(_I0 - _I1).mean() > 50 / 255.
        right_scene = torch.abs(_I1 - _I2).mean() > 50 / 255.
        if left_scene and right_scene:
            t0 = torch.zeros_like(t0).cuda()
            t1 = torch.zeros_like(t0).cuda()
        if left_scene:
            t0 = torch.zeros_like(t0).cuda()
            t1 = torch.ones_like(t1).cuda() * 0.5
        if right_scene:
            t0 = torch.ones_like(t0).cuda() * 0.5
            t1 = torch.zeros_like(t1).cuda()

    t0s = get_timesteps([torch.zeros_like(t0).cuda(), torch.ones_like(t0).cuda()], t0, exp)
    t1s = get_timesteps([torch.zeros_like(t1).cuda(), torch.ones_like(t1).cuda()], t1, exp)

    # 计算i0,i1的中间帧i01, ...
    I01s = []
    I12s = []
    for t0 in t0s:
        I01 = ifnet(torch.cat((_I0, _I1), 1), t0, scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
        I01s.append(I01)
    for t1 in t1s:
        I12 = ifnet(torch.cat((_I1, _I2), 1), t1, scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
        I12s.append(I12)

    I01s = list(map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), I01s))
    I12s = list(map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), I12s))
    _I0, _I1, _I2 = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8),
                        [_I0, _I1, _I2])

    return [_I0] + I01s + [_I1] + I12s + [_I2]


video_capture = cv2.VideoCapture(input)
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

cnt = 1
# start inference
i0 = get()
I0 = load_image(i0, scale)

while True:
    i1 = get()
    if i1 is None:
        break
    i2 = get()
    I1, I2 = load_image(i1, scale), load_image(i2, scale)

    output = make_inference(I0, I1, I2, scale)
    output = output[:-1]
    for x in output:
        put(x)

    i0, I0 = i2, I2
    pbar.update(2)

# 尾
put(i2)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
