# for study only
import math
import subprocess
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import torch
import numpy as np
import time
from models.model_nb222.GMFSS import Model
from models.model_nb222.softsplat import softsplat as warp
import warnings

warnings.filterwarnings("ignore")

input = r'E:\test.mp4'  # input video path
output = r'D:\tmp\output.mkv'  # output video path
scale = 1.0  # flow scale
dst_fps = 60  # target fps (at least greater than source video fps)
global_size = (1920, 1080)  # frame output resolution
hwaccel = True  # Use hardware acceleration video encoder


# deprecated
# swap_thresh means Threshold for applying the swap mask.
# 0 means fully apply the swap mask.
# 0.n means enable swapping when the timestep difference is greater than 0.n.
# 1 means never apply the swap mask.
# swap_thresh = 1

class TMapper:
    def __init__(self, src=-1., dst=0., times=None):
        self.times = dst / src if times is None else times
        self.now_step = -1

    def get_range_timestamps(self, _min: float, _max: float, lclose=True, rclose=False, normalize=True) -> list:
        _min_step = math.ceil(_min * self.times)
        _max_step = math.ceil(_max * self.times)
        _start = _min_step if lclose else _min_step + 1
        _end = _max_step if not rclose else _max_step + 1
        if _start >= _end:
            return []
        if normalize:
            return [((_i / self.times) - _min) / (_max - _min) for _i in range(_start, _end)]
        return [_i / self.times for _i in range(_start, _end)]


def generate_frame_renderer(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)
    read_fps = video_capture.get(cv2.CAP_PROP_FPS)
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{dst_fps}',
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

model = Model()
model.load_model(r'weights\train_log_nb222', -1)
model.device()
model.eval()


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
def make_inference(_I0, _I1, _I2, minus_t, zero_t, plus_t, _scale):
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

    for t in minus_t:
        t = -t
        output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                          timestep1=1 - (t * 2) * drm01))
    for _ in zero_t:
        output1.append(_I1)
    for t in plus_t:
        output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                          timestep1=1 - (t * 2) * drm21))

    _output = output1 + output2
    _output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), _output)

    return _output


video_capture = cv2.VideoCapture(input)
src_fps = video_capture.get(cv2.CAP_PROP_FPS)
assert dst_fps > src_fps, 'dst fps should be greater than src fps'
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

# start inference
i0, i1 = get(), get()
I0, I1 = load_image(i0, scale), load_image(i1, scale)

offset = (dst_fps - src_fps) / dst_fps / 2
t_mapper = TMapper(src_fps, dst_fps)
idx = -1


def calc_t(_last: np.ndarray, _idx: int):
    ori_timestamp = np.array(t_mapper.get_range_timestamps(_idx, _idx + 1, lclose=False, rclose=True, normalize=False))
    timestamp = ori_timestamp + offset
    vfi_timestamp = np.round(timestamp - (_idx + 1), 4)  # head只需要大于等于0的部分, tail只需要小于等于0的部分

    vfi_timestamp = np.concatenate((_last, vfi_timestamp))
    _last = vfi_timestamp[vfi_timestamp > 0.5] - 1
    vfi_timestamp = vfi_timestamp[vfi_timestamp <= 0.5]

    minus_t = vfi_timestamp[vfi_timestamp < 0]
    zero_t = vfi_timestamp[vfi_timestamp == 0]
    plus_t = vfi_timestamp[vfi_timestamp > 0]
    return minus_t, zero_t, plus_t, _last


# head
mt, zt, pt, last = calc_t(np.array([]), idx)
output = make_inference(I0, I0, I1, mt, zt, pt, scale)
for x in output:
    put(x)
pbar.update(1)

while True:
    i2 = get()
    if i2 is None:
        break
    I2 = load_image(i2, scale)

    mt, zt, pt, last = calc_t(last, idx)
    output = make_inference(I0, I1, I2, mt, zt, pt, scale)
    for x in output:
        put(x)

    i0, i1 = i1, i2
    I0, I1 = I1, I2
    pbar.update(1)

# tail(At the end, i0 and i1 have moved to the positions of index -2 and -1 frames.)
mt, zt, pt, last = calc_t(last, idx)
output = make_inference(I0, I1, I1, mt, zt, pt, scale)
for x in output:
    put(x)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()