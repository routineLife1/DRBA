# v2: the workflow is streamlined (but more reliant on transition recognition)
from tqdm import tqdm
import argparse
import time
from models.utils.tools import *
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation a video with DRBA')
    parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='rife',
                        help='model network type, current support rife/gmfss/gmfss_union')
    parser.add_argument('-i', '--input', dest='input', type=str, default='input.mp4',
                        help='absolute path of input video')
    parser.add_argument('-o', '--output', dest='output', type=str, default='output.mp4',
                        help='absolute path of output video')
    parser.add_argument('-fps', '--dst_fps', dest='dst_fps', type=float, default=60, help='interpolate to ? fps')
    parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                        help='enable scene change detection')
    parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                        help='ssim scene detection threshold')
    parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=True,
                        help='enable hardware acceleration encode(require nvidia graph card)')
    parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                        help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
    return parser.parse_args()


def load_model(model_type):
    if model_type == 'rife':
        from models.rife import RIFE

        model = RIFE(weights=r'weights/train_log_rife_426_heavy', scale=scale, device=device)
    elif model_type == 'gmfss':
        from models.gmfss import GMFSS

        model = GMFSS(weights=r'weights/train_log_gmfss', scale=scale, device=device)
    elif model_type == 'gmfss_union':
        from models.gmfss_union import GMFSS_UNION

        model = GMFSS_UNION(weights=r'weights/train_log_gmfss_union', scale=scale, device=device)
    else:
        raise ValueError(f'model_type must in {model_type}')

    return model


def inference():
    video_io = VideoFI_IO(input_path, output_path, dst_fps=dst_fps, times=-1, hwaccel=hwaccel)
    src_fps = video_io.src_fps
    if dst_fps <= src_fps:
        raise ValueError(f'dst fps should be greater than src fps, but got dst_fps={dst_fps} and src_fps={src_fps}')
    pbar = tqdm(total=video_io.total_frames_count)

    # start inference
    i0 = video_io.read_frame()
    size = get_valid_net_inp_size(i0, model.scale, div=model.pad_size)
    src_size, dst_size = size['src_size'], size['dst_size']

    I0 = to_inp(i0, dst_size)

    t_mapper = TMapper(src_fps, dst_fps)
    idx = 0

    def calc_t(_idx: float):
        timestamp = np.array(
            t_mapper.get_range_timestamps(_idx, _idx + 2, lclose=True, rclose=False, normalize=False))
        vfi_timestamp = np.round(timestamp - _idx, 4)  # [0, 2)

        return vfi_timestamp

    while True:
        i1 = video_io.read_frame()
        if i1 is None:
            break
        i2 = video_io.read_frame()
        if i2 is None:
            break
        I1 = to_inp(i1, dst_size)
        I2 = to_inp(i2, dst_size)

        left_scene = check_scene(I0, I1, scdet_threshold) if enable_scdet else False
        right_scene = check_scene(I1, I2, scdet_threshold) if enable_scdet else False

        ts = calc_t(idx)
        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if left_scene and right_scene:  # scene transition occurs at I0~I1, also occurs at I1~I2
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1]
            output = [I0 for _ in left_ts]
            output.extend([I1 for _ in right_ts])

        elif left_scene and not right_scene:  # scene transition occurs at I0~I1
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1] - 1

            output = [I0 for _ in left_ts]
            output.extend(model.inference_ts(I1, I2, right_ts))

        elif not left_scene and right_scene:  # scene transition occurs at I1~I2
            left_ts = ts[ts <= 1]
            right_ts = ts[ts > 1] - 1

            output = model.inference_ts(I0, I1, left_ts)
            output.extend([I1 for _ in right_ts])

        else:
            # v2 no longer need reuse
            output, _ = model.inference_ts_drba(I0, I1, I2, ts, _reuse=None)

        # debug
        # for i in range(len(output)):
        #     output[i] = mark_tensor(output[i], f"{ts[i] + idx}")

        for x in output:
            video_io.write_frame(to_out(x, src_size))

        I0 = I2
        idx += 2
        pbar.update(2)

    # tail (if exist)
    if i1 is not None:
        I1 = to_inp(i1, dst_size)
        ts = t_mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=True, normalize=False)
        scene = check_scene(I0, I1, scdet_threshold) if enable_scdet else False

        if scene:
            output = [I0 for _ in ts]
        else:
            output = model.inference_ts(I0, I1, ts)

        for x in output:
            video_io.write_frame(to_out(x, src_size))

    idx += 1
    pbar.update(1)

    # wait for output
    while not video_io.finish_writing():
        time.sleep(1)
    pbar.close()


if __name__ == '__main__':
    args = parse_args()
    model_type = args.model_type  # model network type
    input_path = args.input  # input video path
    output_path = args.output  # output video path
    scale = args.scale  # flow scale
    dst_fps = args.dst_fps  # Must be an integer multiple
    enable_scdet = args.enable_scdet  # enable scene change detection
    scdet_threshold = args.scdet_threshold  # scene change detection threshold
    hwaccel = args.hwaccel  # Use hardware acceleration video encoder

    model = load_model(model_type)
    inference()
