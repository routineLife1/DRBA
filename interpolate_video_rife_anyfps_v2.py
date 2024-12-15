# for real-time playback(+TensorRT)
import math
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import subprocess
import argparse
import torch
import numpy as np
import time
from models.pytorch_msssim import ssim_matlab
from torch.nn import functional as F
from models.rife_426_heavy.IFNet_HDv3 import IFNet
import warnings

warnings.filterwarnings("ignore")

HAS_CUDA = True
try:
    import cupy

    if cupy.cuda.get_cuda_path() == None:
        HAS_CUDA = False
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from models.softsplat.softsplat import softsplat as warp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from models.softsplat.softsplat_torch import softsplat as warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Interpolation a video with AFI-ForwardDeduplicate')
parser.add_argument('-i', '--input', dest='input', type=str, default='input.mp4', help='absolute path of input video')
parser.add_argument('-o', '--output', dest='output', type=str, default='output.mp4',
                    help='absolute path of output video')
parser.add_argument('-fps', '--dst_fps', dest='dst_fps', type=float, default=60, help='interpolate to ? fps')
parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=True,
                    help='enable scene change detection')
parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                    help='ssim scene detection threshold')
parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=True,
                    help='enable hardware acceleration encode(require nvidia graph card)')
parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                    help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
args = parser.parse_args()

input = args.input  # input video path
output = args.output  # output video path
scale = args.scale  # flow scale
dst_fps = args.dst_fps  # Must be an integer multiple
enable_scdet = args.enable_scdet  # enable scene change detection
scdet_threshold = args.scdet_threshold  # scene change detection threshold
hwaccel = args.hwaccel  # Use hardware acceleration video encoder


def check_scene(x1, x2):
    if not enable_scdet:
        return False
    x1 = F.interpolate(x1, (32, 32), mode='bilinear', align_corners=False)
    x2 = F.interpolate(x2, (32, 32), mode='bilinear', align_corners=False)
    return ssim_matlab(x1, x2) < scdet_threshold


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


video_capture = cv2.VideoCapture(input)
width, height = map(int, map(video_capture.get, [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]))


def generate_frame_renderer(input_path, output_path):
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{dst_fps}',
        '-s', f'{width}x{height}',
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


ifnet = IFNet().to(device).eval()
ifnet.load_state_dict(convert(torch.load(r'weights/train_log_rife_426_heavy/flownet.pkl', map_location='cpu')), False)
flownet = ifnet


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.


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
    while ret is True:
        r_buffer.put(__x)
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    global ffmpeg_writer
    while True:
        item = w_buffer.get()
        if item is None:
            break
        result = cv2.resize(item, (width, height))
        ffmpeg_writer.stdin.write(np.ascontiguousarray(result[:, :, ::-1]))
    ffmpeg_writer.stdin.close()
    ffmpeg_writer.wait()


@torch.inference_mode()
@torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
def make_inference(_I0, _I1, _I2, minus_t, zero_t, plus_t, _left_scene, _right_scene, _scale):
    def calc_flow(model, a, b):
        imgs = torch.cat((a, b), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        # get top scale flow flow0.5 -> 0/1
        flow = model(imgs, timestep=0.5, scale_list=scale_list)[1][-1]
        flow50, flow51 = flow[:, :2], flow[:, 2:]

        # only need forward direction flow
        flow05_primary = warp(flow51, flow50, None, 'avg')
        flow15_primary = warp(flow50, flow51, None, 'avg')

        # qvi
        # flow05, norm2 = fwarp(flow50, flow50)
        # flow05[norm2]...
        # flow05 = -flow05

        flow05_secondary = -warp(flow50, flow50, None, 'avg')
        flow15_secondary = -warp(flow51, flow51, None, 'avg')

        _flow01_primary = flow05_primary * 2
        _flow10_primary = flow15_primary * 2

        _flow01_secondary = flow05_secondary * 2
        _flow10_secondary = flow15_secondary * 2

        return _flow01_primary, _flow10_primary, _flow01_secondary, _flow10_secondary

    # Flow distance calculator
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    flow10_p, flow01_p, flow01_s, flow10_s = calc_flow(flownet, _I1, _I0)
    flow12_p, flow21_p, flow12_s, flow21_s = calc_flow(flownet, _I1, _I2)

    # Compute the distance using the optical flow and distance calculator
    d10_p = distance_calculator(flow10_p) + 1e-4
    d12_p = distance_calculator(flow12_p) + 1e-4
    d10_s = distance_calculator(flow10_s) + 1e-4
    d12_s = distance_calculator(flow12_s) + 1e-4

    # Calculate the distance ratio map
    drm10_p = d10_p / (d10_p + d12_p)
    drm12_p = d12_p / (d10_p + d12_p)
    drm10_s = d10_s / (d10_s + d12_s)
    drm12_s = d12_s / (d10_s + d12_s)

    ones_mask = torch.ones_like(drm10_p, device=drm10_p.device)

    def get_drm_t(drm, _t, precision=1e-3):
        _x, b = 0.5, 0.5
        l, r = 0, 1

        x_drm, b_drm = drm.clone(), drm.clone()
        l_drm, r_drm = torch.zeros_like(x_drm, device=x_drm.device), torch.ones_like(x_drm, device=x_drm.device)

        while abs(_x - _t) > precision:
            if _x > _t:
                r = _x
                _x = _x - (_x - l) * b

                r_drm = x_drm.clone()
                x_drm = x_drm - (x_drm - l_drm) * b_drm

            if _x < _t:
                l = _x
                _x = _x + (r - _x) * b

                l_drm = x_drm.clone()
                x_drm = x_drm + (r_drm - x_drm) * b_drm

        return x_drm

    def calc_drm_rife(_t):
        # The distance ratio map (drm) is initially aligned with I1.
        # To align it with I0 and I2, we need to warp the drm maps.
        # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
        # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
        drm10r_p = get_drm_t(1 - drm10_p, _t)
        drm12r_p = get_drm_t(1 - drm12_p, _t)
        drm10r_s = get_drm_t(1 - drm10_s, _t)
        drm12r_s = get_drm_t(1 - drm12_s, _t)
        _drm01r_p = warp(drm10r_p, flow10_p * drm10r_p, None, strMode='avg')
        _drm21r_p = warp(drm12r_p, flow12_p * drm12r_p, None, strMode='avg')
        _drm01r_s = warp(drm10r_s, flow10_s * drm10r_s, None, strMode='avg')
        _drm21r_s = warp(drm12r_s, flow12_s * drm12r_s, None, strMode='avg')

        warped_ones_mask01r_p = warp(ones_mask, flow10_p * drm10r_p, None, strMode='avg')
        warped_ones_mask21r_p = warp(ones_mask, flow12_p * drm12r_p, None, strMode='avg')
        warped_ones_mask01r_s = warp(ones_mask, flow10_s * drm10r_s, None, strMode='avg')
        warped_ones_mask21r_s = warp(ones_mask, flow12_s * drm12r_s, None, strMode='avg')

        holes01r_p = warped_ones_mask01r_p < 0.999
        holes21r_p = warped_ones_mask21r_p < 0.999

        _drm01r_p[holes01r_p] = _drm01r_s[holes01r_p]
        _drm21r_p[holes21r_p] = _drm21r_s[holes21r_p]

        holes01r_s = warped_ones_mask01r_s < 0.999
        holes21r_s = warped_ones_mask21r_s < 0.999

        holes01r = torch.logical_and(holes01r_p, holes01r_s)
        holes21r = torch.logical_and(holes21r_p, holes21r_s)

        _drm01r_p[holes01r] = drm10r_p[holes01r]
        _drm21r_p[holes21r] = drm12r_p[holes21r]

        _drm01r_p, _drm21r_p = map(lambda x: torch.nn.functional.interpolate(x, size=_I0.shape[2:], mode='bilinear',
                                                                             align_corners=False),
                                   [_drm01r_p, _drm21r_p])

        return _drm01r_p, _drm21r_p

    output1, output2 = list(), list()

    if _left_scene:
        for _ in minus_t:
            zero_t = np.append(zero_t, 0)
        minus_t = list()

    if _right_scene:
        for _ in plus_t:
            zero_t = np.append(zero_t, 0)
        plus_t = list()

    disable_drm = False
    if (_left_scene and not _right_scene) or (not _left_scene and _right_scene):
        drm01r, drm21r = (ones_mask.clone() * 0.5 for _ in range(2))
        drm01r, drm21r = map(lambda x: torch.nn.functional.interpolate(x, size=_I0.shape[2:], mode='bilinear',
                                                                       align_corners=False), [drm01r, drm21r])
        disable_drm = True

    for t in minus_t:
        t = -t
        if t == 1:
            output1.append(_I0)
            continue
        if not disable_drm:
            drm01r, _ = calc_drm_rife(t)
        output1.append(ifnet(torch.cat((_I1, _I0), 1), timestep=drm01r,
                             scale_list=[16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale])[0])

    for _ in zero_t:
        output1.append(_I1)

    for t in plus_t:
        if t == 1:
            output2.append(_I2)
            continue
        if not disable_drm:
            _, drm21r = calc_drm_rife(t)
        output2.append(ifnet(torch.cat((_I1, _I2), 1), timestep=drm21r,
                             scale_list=[16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale])[0])

    _output = output1 + output2

    _output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), _output)

    return _output


src_fps = video_capture.get(cv2.CAP_PROP_FPS)
assert dst_fps > src_fps, 'dst fps should be greater than src fps'
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

# start inference
i0 = get()
I0 = load_image(i0, scale)

t_mapper = TMapper(src_fps, dst_fps)
idx = 0


def calc_t(_idx: float):
    timestamp = np.array(
        t_mapper.get_range_timestamps(_idx, _idx + 2, lclose=True, rclose=False, normalize=False))
    vfi_timestamp = np.round(timestamp - _idx, 4) - 1

    minus_t = vfi_timestamp[vfi_timestamp < 0]
    zero_t = vfi_timestamp[vfi_timestamp == 0]
    plus_t = vfi_timestamp[vfi_timestamp > 0]
    return minus_t, zero_t, plus_t


while True:
    i1 = get()
    if i1 is None:
        break
    i2 = get()
    if i2 is None:
        break
    I1 = load_image(i1, scale)
    I2 = load_image(i2, scale)

    left_scene = check_scene(I0, I1)
    right_scene = check_scene(I1, I2)

    mt, zt, pt = calc_t(idx)
    output = make_inference(I0, I1, I2, mt, zt, pt, left_scene, right_scene, scale)
    for x in output:
        put(x)

    i0 = i2
    I0 = I2
    idx += 2
    pbar.update(2)

output = []
# tail
if i1 is not None:
    I1 = load_image(i1, scale)
    ts = t_mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=True, normalize=False)
    for t in ts:
        if 0 < t < 1:
            output.append(ifnet(torch.cat((I0, I1), 1), timestep=t,
                                scale_list=[16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale])[0])
        else:
            output.append(I0 if t == 0 else I1)

    output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), output)

    for x in output:
        put(x)
put(None)
idx += 1
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
