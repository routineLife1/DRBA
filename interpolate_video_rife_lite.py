# for real-time playback(+TensorRT)
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import subprocess
import torch
import numpy as np
import time
from models.model_nb222.MetricNet import MetricNet
from models.model_nb222.softsplat import softsplat as warp
from models.rife_422_lite.IFNet_HDv3 import IFNet
from models.FastFlowNet.models.FastFlowNet_v2 import FastFlowNet
import warnings

warnings.filterwarnings("ignore")

input = r'E:\01.mkv'  # input video path
output = r'D:\tmp\output.mkv'  # output video path
scale = 1.0  # flow scale
times = 5  # Must be an integer multiple
global_size = (1920, 1080)  # frame output resolution
hwaccel = True  # Use hardware acceleration video encoder


def generate_frame_renderer(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)
    read_fps = video_capture.get(cv2.CAP_PROP_FPS)
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{read_fps * times}',
        '-s', f'{global_size[0]}x{global_size[1]}',
        '-i', 'pipe:0', '-i', input_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', encoder, "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-qp", "16", '-preset', preset,
        '-c:a', 'aac', '-b:a', '320k', f'{output_path}'
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


ffmpeg_writer = generate_frame_renderer(input, output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


ifnet = IFNet().cuda().eval()
ifnet.load_state_dict(convert(torch.load(r'weights\train_log_rife_422_lite\flownet.pkl', map_location='cpu')), False)
flownet = FastFlowNet().cuda().eval()
flownet.load_state_dict(torch.load(r'weights\train_log_rife_422_lite\fastflownet_ft_sintel.pth', map_location='cpu'))
metricnet = MetricNet().cuda().eval()
metricnet.load_state_dict(torch.load(r'weights\train_log_rife_422_lite\metric.pkl', map_location='cpu'))


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.


def load_image(img, _scale):
    h, w, _ = img.shape
    while h * _scale % 64 != 0:
        h += 1
    while w * _scale % 64 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    img = to_tensor(img)
    return img


def put(things):
    write_buffer.put(things)


def get():
    return read_buffer.get()


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    global ffmpeg_writer
    while True:
        item = w_buffer.get()
        if item is None:
            break
        result = cv2.resize(item, global_size)
        ffmpeg_writer.stdin.write(np.ascontiguousarray(result[:, :, ::-1]))
    ffmpeg_writer.stdin.close()
    ffmpeg_writer.wait()


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

    # Flow distance calculator
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    # When using FastFlowNet to calculate optical flow, the input image size is uniformly reduced to half of the original size.
    # FastFlowNet requires the input image dimensions to be divisible by 64.
    I0f, I1f, I2f = map(
        lambda x: torch.nn.functional.interpolate(x, size=(576, 1024), mode='bilinear', align_corners=False),
        [_I0, _I1, _I2])

    flow01 = calc_flow(flownet, I0f, I1f)
    flow10 = calc_flow(flownet, I1f, I0f)
    flow12 = calc_flow(flownet, I1f, I2f)
    flow21 = calc_flow(flownet, I2f, I1f)

    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    I0ff, I1ff, I2ff = map(
        lambda x: torch.nn.functional.interpolate(x, size=flow01.shape[2:], mode='bilinear', align_corners=False),
        [_I0, _I1, _I2])

    _, metric10 = metricnet(I0ff, I1ff, flow01, flow10)
    metric12, _ = metricnet(I1ff, I2ff, flow12, flow21)

    ones_mask = torch.ones_like(drm10, device=drm10.device)

    def calc_drm_rife(_t):
        # The distance ratio map (drm) is initially aligned with I1.
        # To align it with I0 and I2, we need to warp the drm maps.
        # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
        # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
        drm01r = warp(1 - drm10, flow10 * ((1 - drm10) * 2) * _t, metric10, strMode='soft')
        drm21r = warp(1 - drm12, flow12 * ((1 - drm12) * 2) * _t, metric12, strMode='soft')

        warped_ones_mask01r = warp(ones_mask, flow10 * ((1 - drm01r) * 2) * _t, metric10, strMode='soft')
        warped_ones_mask21r = warp(ones_mask, flow12 * ((1 - drm21r) * 2) * _t, metric12, strMode='soft')

        holes01r = warped_ones_mask01r < 0.999
        holes21r = warped_ones_mask21r < 0.999

        drm01r[holes01r] = (1 - drm10)[holes01r]
        drm21r[holes21r] = (1 - drm12)[holes21r]

        drm01r, drm21r = map(lambda x: torch.nn.functional.interpolate(x, size=_I0.shape[2:], mode='bilinear',
                                                                       align_corners=False), [drm01r, drm21r])
        return drm01r, drm21r

    output1, output2 = list(), list()
    _output = list()
    if times % 2:
        for i in range((times - 1) // 2):
            t = (i + 1) / times
            # Adjust timestep parameters for interpolation between frames I0, I1, and I2
            # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2

            drm01r, drm21r = calc_drm_rife(t)
            I10 = ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
            I12 = ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])

            output1.append(I10)
            output2.append(I12)
        _output = list(reversed(output1)) + [_I1] + output2
    else:
        for i in range(times // 2):
            t = (i + 0.5) / times

            drm01r, drm21r = calc_drm_rife(t)
            I10 = ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
            I12 = ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])

            output1.append(I10)
            output2.append(I12)
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

# start inference
i0, i1 = get(), get()
I0, I1 = load_image(i0, scale), load_image(i1, scale)

# head
output = make_inference(I0, I0, I1, scale)
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

# tail(At the end, i0 and i1 have moved to the positions of index -2 and -1 frames.)
output = make_inference(I0, I1, I1, scale)
for x in output:
    put(x)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
