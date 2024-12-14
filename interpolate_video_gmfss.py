# for study only
import subprocess
import argparse
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import torch
import numpy as np
import time
from models.model_nb222.GMFSS import Model
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
parser.add_argument('-t', '--times', dest='times', type=int, default=2, help='interpolate to ?x fps')
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
times = args.times  # Must be an integer multiple
enable_scdet = args.enable_scdet  # enable scene change detection
scdet_threshold = args.scdet_threshold  # scene change detection threshold
hwaccel = args.hwaccel  # Use hardware acceleration video encoder

# deprecated
# swap_thresh means Threshold for applying the swap mask.
# 0 means fully apply the swap mask.
# 0.n means enable swapping when the timestep difference is greater than 0.n.
# 1 means never apply the swap mask.
# swap_thresh = 1

video_capture = cv2.VideoCapture(input)
read_fps = video_capture.get(cv2.CAP_PROP_FPS)
width, height = map(int, map(video_capture.get, [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]))


def generate_frame_renderer(input_path, output_path):
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{read_fps * times}',
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

model = Model()
model.load_model(r'weights\train_log_nb222', -1)
model.device()
model.eval()


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
    while ret:
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
def make_inference(_I0, _I1, _I2, _scale, _reuse=None):
    # Flow distance calculator
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    reuse_i1i0 = model.reuse(_I1, _I0, scale) if _reuse is None else _reuse
    reuse_i1i2 = model.reuse(_I1, _I2, scale)

    flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
    flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]

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
            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=t * (2 * (1 - drm10)),
                                              timestep1=1 - (t * 2) * drm01))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=t * (2 * (1 - drm12)),
                                              timestep1=1 - (t * 2) * drm21))
        _output = list(reversed(output1)) + [_I1] + output2
    else:
        for i in range(times // 2):
            t = (i + 0.5) / times
            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=t * (2 * (1 - drm10)),
                                              timestep1=1 - (t * 2) * drm01))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=t * (2 * (1 - drm12)),
                                              timestep1=1 - (t * 2) * drm21))
        _output = list(reversed(output1)) + output2

    _output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), _output)

    # next reuse_i1i0 = reverse(current reuse_i1i2)
    # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
    # _reuse = (f1, f0, m1, m0, feat1, feat0)
    _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]

    return _output, _reuse


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
output, reuse = make_inference(I0, I0, I1, scale, None)
for x in output:
    put(x)
pbar.update(1)

while True:
    i2 = get()
    if i2 is None:
        break
    I2 = load_image(i2, scale)

    output, reuse = make_inference(I0, I1, I2, scale, reuse)
    for x in output:
        put(x)

    i0, i1 = i1, i2
    I0, I1 = I1, I2
    pbar.update(1)

# tail(At the end, i0 and i1 have moved to the positions of index -2 and -1 frames.)
output, _ = make_inference(I0, I1, I1, scale, reuse)
for x in output:
    put(x)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
