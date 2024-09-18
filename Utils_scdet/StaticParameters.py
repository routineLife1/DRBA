import enum
import json
import os
import platform
import re
import threading

import numpy as np

abspath = os.path.abspath(__file__)
appDir = os.path.dirname(os.path.dirname(abspath))
abs_path_to_vspipe_trt = os.path.join(appDir, "vspipe/vapoursynth64/coreplugins/vsmlrt-cuda")  # trt dll env using vsmlrt-cuda
abs_path_to_torch_lib = os.path.join(appDir, "torch/lib")  # trt dll env using vsmlrt-cuda

INVALID_CHARACTERS = ["'", '"', '“', '”', '%']
IS_RELEASE = False
TB_LIMIT = 0 if IS_RELEASE else None  # Traceback Limit
PATH_LEN_LIMIT = 230
MAX_FRAME_CNT = int(10 ** 10)

PREVIEW_INTERVAL = 0.5

IS_WINDOW = 'window' in platform.platform().lower()
IS_DARWIN = 'Darwin' in platform.system()
IS_CLI = not IS_WINDOW
OLS_PATH = os.path.join(appDir, "one_line_shot_args.exe" if IS_WINDOW else "one_line_shot_args")
IS_DEBUG = not os.path.exists(OLS_PATH)

DEBUG_VFI = False
DEBUG_SR = False
DEBUG_RESTORE = False
TASK_TIMEOUT = 3600 if not DEBUG_VFI else 10


class RELEASE_TYPE(enum.Enum):
    RELEASE = 0
    ALPHA = 1
    BETA = 2
    DEMO = 3
    SAE = 4

    @staticmethod
    def get_release_type() -> 'RELEASE_TYPE':
        release_info_path = os.path.join(appDir, "release_info.json")
        if not os.path.exists(release_info_path):
            return RELEASE_TYPE.RELEASE
        with open(release_info_path, "r") as f:
            release_info = json.load(f)
        return RELEASE_TYPE(release_info["release_type"])


class TASKBAR_STATE(enum.Enum):
    TBPF_NOPROGRESS = 0x00000000
    TBPF_INDETERMINATE = 0x00000001
    TBPF_NORMAL = 0x00000002
    TBPF_ERROR = 0x00000004
    TBPF_PAUSED = 0x00000008


class HDR_STATE(enum.Enum):
    AUTO = -2
    NOT_CHECKED = -1
    NONE = 0
    CUSTOM_HDR = 1
    HDR10 = 2
    HDR10_PLUS = 3
    DOLBY_VISION = 4
    HLG = 5


class RESIZE_INDEX(enum.Enum):
    CUSTOM = 0
    R100 = 1
    R200 = 2
    R300 = 3
    R400 = 4
    R050 = 5
    R025 = 6
    SD480 = 7
    HD720p = 8
    HD1080p = 9
    UHD4K = 10
    UHD8K = 11

    @staticmethod
    def update_resize_params(index, frame_size: [int, int]) -> [int, int]:
        """Get Updated Resize Params

        :param index:
        :param frame_size: Input Video Frame Size
        :return:
        """
        if not any(frame_size):
            return frame_size  # (0,0)
        ratio = frame_size[0] / frame_size[1]
        if index == RESIZE_INDEX.CUSTOM:
            return frame_size
        elif index == RESIZE_INDEX.R100:
            return frame_size
        elif index == RESIZE_INDEX.R200:
            frame_size = int(frame_size[0] * 2.0), int(frame_size[1] * 2.0)
        elif index == RESIZE_INDEX.R300:
            frame_size = int(frame_size[0] * 3.0), int(frame_size[1] * 3.0)
        elif index == RESIZE_INDEX.R400:
            frame_size = int(frame_size[0] * 4.0), int(frame_size[1] * 4.0)
        elif index == RESIZE_INDEX.R050:
            frame_size = int(frame_size[0] * 0.5), int(frame_size[1] * 0.5)
        elif index == RESIZE_INDEX.R025:
            frame_size = int(frame_size[0] * 0.25), int(frame_size[1] * 0.25)
        elif index == RESIZE_INDEX.SD480:
            frame_size = int(480 * ratio), 480
        elif index == RESIZE_INDEX.HD720p:
            frame_size = int(720 * ratio), 720
        elif index == RESIZE_INDEX.HD1080p:
            frame_size = int(1080 * ratio), 1080
        elif index == RESIZE_INDEX.UHD4K:
            frame_size = int(2160 * ratio), 2160
        elif index == RESIZE_INDEX.UHD8K:
            frame_size = int(4320 * ratio), 4320
        w = frame_size[0] + 1 if frame_size[0] % 2 == 1 else frame_size[0]
        h = frame_size[1] + 1 if frame_size[1] % 2 == 1 else frame_size[1]
        return w, h

    @staticmethod
    def get_regular_sr_ratio(index) -> float:
        """Return Regular Sr Ratio for Sr._get_sr_scale()

        :param index:
        :return:
        """
        if index == RESIZE_INDEX.R100:
            return 1.0
        elif index == RESIZE_INDEX.R200:
            return 2.0
        elif index == RESIZE_INDEX.R300:
            return 3.0
        elif index == RESIZE_INDEX.R400:
            return 4.0
        elif index == RESIZE_INDEX.R050:
            return 0.5
        elif index == RESIZE_INDEX.R025:
            return 0.25
        else:
            return 1.0


class RT_RATIO(enum.Enum):
    """
    Resolution Transfer Ratio
    """
    AUTO = 0
    WHOLE = 1
    TWO_THIRDS = 2
    HALF = 3
    QUARTER = 4

    @staticmethod
    def get_auto_transfer_ratio(sr_times: float):
        if sr_times >= 1:
            return RT_RATIO.WHOLE
        elif 0.6 <= sr_times < 1:
            return RT_RATIO.TWO_THIRDS
        elif 0.5 <= sr_times < 0.75:
            return RT_RATIO.HALF
        else:
            return RT_RATIO.QUARTER

    @staticmethod
    def get_surplus_sr_scale(scale: float, ratio):
        if ratio == RT_RATIO.WHOLE:
            return scale
        elif ratio == RT_RATIO.TWO_THIRDS:
            return scale * 1.5
        elif ratio == RT_RATIO.HALF:
            return scale * 2
        elif ratio == RT_RATIO.QUARTER:
            return scale * 4
        else:
            return scale

    @staticmethod
    def get_modified_resolution(params: tuple, ratio, is_reverse=False, keep_single=False):
        w, h = params
        mod_ratio = 1
        if ratio == RT_RATIO.WHOLE:
            mod_ratio = 1
        elif ratio == RT_RATIO.TWO_THIRDS:
            mod_ratio = 2 / 3
        elif ratio == RT_RATIO.HALF:
            mod_ratio = 0.5
        elif ratio == RT_RATIO.QUARTER:
            mod_ratio = 0.25
        if not is_reverse:
            w, h = int(w * mod_ratio), int(h * mod_ratio)
        else:
            w, h = int(w / mod_ratio), int(h / mod_ratio)
        if not keep_single:
            if abs(w) % 2:
                w += 1
            if abs(h) % 2:
                h += 1
        return w, h


class SR_TILESIZE_STATE(enum.Enum):
    NONE = 0
    CUSTOM = 1
    VRAM_2G = 2
    VRAM_4G = 3
    VRAM_6G = 4
    VRAM_8G = 5
    VRAM_12G = 6

    @staticmethod
    def get_tilesize(state):
        if state == SR_TILESIZE_STATE.NONE:
            return 0
        if state == SR_TILESIZE_STATE.VRAM_2G:
            return 100
        if state == SR_TILESIZE_STATE.VRAM_4G:
            return 200
        if state == SR_TILESIZE_STATE.VRAM_6G:
            return 1000
        if state == SR_TILESIZE_STATE.VRAM_8G:
            return 1200
        if state == SR_TILESIZE_STATE.VRAM_12G:
            return 2000
        return 100


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov', '.gif', '.webp']


class EncodePresetAssemply:
    """
    All encoders must have `H264,8bit`
    Auto Settings System will use H265,10bit for HDR input
    """
    encoder = {  # the order should correspond to "slow fast medium" - "3 1 2" out of "0 1 2 3 4", at least 3
        "AUTO": {"AUTO": ["AUTO"]},
        "CPU": {
            "H264,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "H265,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "H265,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "AV1,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "AV1,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"],
             "H265,8bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"],
             "H265,10bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },
        "VCE":
            {"H264,8bit": ["quality", "speed", "balanced"],
             "H265,8bit": ["quality", "speed", "balanced"], },
        "NVENCC":
            {"H264,8bit": ["quality", "performance", "default"],
             "H265,8bit": ["quality", "performance", "default"],
             "H265,10bit": ["quality", "performance", "default"],
             "AV1,8bit": ["quality", "performance", "default"],
             "AV1,10bit": ["quality", "performance", "default"], },
        "VCEENCC":
            {"H264,8bit": ["slow", "fast", "balanced"],
             "H265,8bit": ["slow", "fast", "balanced"],
             "H265,10bit": ["slow", "fast", "balanced"],
             "AV1,8bit": ["slower", "fast", "balanced", "slow"],
             "AV1,10bit": ["slower", "fast", "balanced", "slow"], },
        "QSVENCC":
            {"H264,8bit": ["best", "fast", "balanced", "higher", "high", "faster", "fastest"],
             "H265,8bit": ["best", "fast", "balanced", "higher", "high", "faster", "fastest"],
             "H265,10bit": ["best", "fast", "balanced", "higher", "high", "faster", "fastest"],
             "AV1,8bit": ["best", "fast", "balanced", "higher", "high", "faster", "fastest"],
             "AV1,10bit": ["best", "fast", "balanced", "higher", "high", "faster", "fastest"], },
        # "SVT":
        #     {"VP9,8bit": ["slowest", "slow", "fast", "faster"],
        #      "H265,8bit": ["slowest", "slow", "fast", "faster"],
        #      "AV1,8bit": ["slowest", "slow", "fast", "faster"],
        #      },
    }
    ffmpeg_encoders = ["AUTO", "CPU", "NVENC", "QSV", "VCE"]
    darwin_encoders = ["AUTO", "CPU"]
    encc_encoders = ["NVENCC", "QSVENCC", "VCEENCC"]
    community_encoders = ffmpeg_encoders if not IS_DARWIN else darwin_encoders
    params_libx265s = {
        "fast": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=1:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightb=1:strong-intra-smoothing=0:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=15:lookahead-slices=8:b-adapt=1:bframes=4:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
        "fast_FD+ZL": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=0:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightp=0:strong-intra-smoothing=0:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=8:b-adapt=0:bframes=0:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
        "slow": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=1:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightb=1:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=35:lookahead-slices=4:b-adapt=2:bframes=6:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
        "slow_FD+ZL": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=0:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightp=0:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=4:b-adapt=0:bframes=0:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
    }

    params_libx264s = {
        "fast": "keyint=250:min-keyint=1:bframes=3:b-adapt=1:open-gop=0:ref=2:rc-lookahead=20:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=1:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock='-1:-1'",
        "fast_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=2:rc-lookahead=25:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=0:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock=false:cabac=0:weightb=0",
        "slow": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=8:rc-lookahead=35:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=2:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock='-1:-1'",
        "slow_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=8:rc-lookahead=25:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=0:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock=false:cabac=0:weightb=0",
    }

    h265_hdr10_info = "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,100:hdr10-opt=1:repeat-headers=1"  # no need to transfer color metadatas(useless)
    h264_hdr10_info = "mastering-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'"
    master_display_info = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    max_cll_info = '1000,100'

    class HdrInfo:  # Display P3
        G: tuple = (13250, 34500)
        B: tuple = (7500, 3000)
        R: tuple = (34000, 16000)
        WP: tuple = (15635, 16450)
        LL: tuple = (10000000, 50)
        CLL: tuple = (1000, 100)

    @staticmethod
    def update_hdr_master_display(md_info):
        EncodePresetAssemply.master_display_info = f"G({md_info['G'][0]},{md_info['G'][1]})" \
                                                   f"B({md_info['B'][0]},{md_info['B'][1]})" \
                                                   f"R({md_info['R'][0]},{md_info['R'][1]})" \
                                                   f"WP({md_info['WP'][0]},{md_info['WP'][1]})" \
                                                   f"L({md_info['LL'][0]},{md_info['LL'][1]})"
        EncodePresetAssemply.HdrInfo.G = md_info['G'][0], md_info['G'][1]
        EncodePresetAssemply.HdrInfo.B = md_info['B'][0], md_info['B'][1]
        EncodePresetAssemply.HdrInfo.R = md_info['R'][0], md_info['R'][1]
        EncodePresetAssemply.HdrInfo.WP = md_info['WP'][0], md_info['WP'][1]
        EncodePresetAssemply.HdrInfo.LL = md_info['LL'][0], md_info['LL'][1]
        is_update_cll = 'CLL' in md_info
        if is_update_cll:
            EncodePresetAssemply.max_cll_info = f"{md_info['CLL'][0]},{md_info['CLL'][1]}"
            EncodePresetAssemply.HdrInfo.CLL = md_info['CLL'][0], md_info['CLL'][1]
        EncodePresetAssemply.h264_hdr10_info = re.sub("mastering-display='.*?'",
                                                      f"mastering-display='{EncodePresetAssemply.master_display_info}'",
                                                      EncodePresetAssemply.h264_hdr10_info)
        EncodePresetAssemply.h265_hdr10_info = re.sub("master-display=.*?:max-cll=.*?:",
                                                      f"master-display={EncodePresetAssemply.master_display_info}:"
                                                      f"max-cll={EncodePresetAssemply.max_cll_info}:",
                                                      EncodePresetAssemply.h265_hdr10_info)

    @staticmethod
    def get_encoder_format(encoder: str, filter_str: str):
        if encoder not in EncodePresetAssemply.encoder:
            return ["H264,8bit"]
        formats = EncodePresetAssemply.encoder[encoder].keys()
        formats = list(filter(lambda x: filter_str in x, formats))
        return formats


class ColorTag:
    """
    transfer ffmpeg's key to vspipe's or mkvmerge
    ffmpeg: https://ffmpeg.org/ffmpeg-filters.html#colormatrix
    vapoursynth: http://www.vapoursynth.com/doc/functions/video/resize.html
    mkvmerge: https://man.archlinux.org/man/mkvmerge.1

    ffmpeg: color tag -> get_ffmpeg_color_tag_dict -> color cli tag dict
    vspipe: color tag -> ffmpeg_to_vspipe -> triple tag -> update args data
    mkvmerge: color tag -> ffmpeg_to_mkvmerge_color_tag -> mkvmerge color tag -> get_mkvmerge_color_tag_dict ->
              color cli tag dict
    encc: color tag -> get_encc_color_tag_dict -> color cli tag dict
    """
    matrix_f2v_transfer = {'709': '709', '470': '470bg', '601': 'unspec', '170': '170m', '240': '240m',
                           '2020c': '2020cl', '2020nc': '2020ncl', 'fcc': 'fcc', 'ycgco': 'ycgco', 'ictcp': 'ictcp',
                           'derived-nc': 'chromancl', 'derived-c': 'chromacl',
                           'unknown': 'unspec'}
    transf_f2v_transfer = {'709': '709', '470m': '470m', '470bg': '470bg', '601': '601', '240': '240m',
                           '2020': '2020_10', '2084': 'st2084', 'arib': 'std-b67', 'linear': 'linear',
                           'log100': 'log100', 'log316': 'log316',
                           'unknown': 'unspec'}
    primar_f2v_transfer = {'709': '709', '470m': '470m', '470bg': '470bg', '601': '601', '170': '170m', '240': '240m',
                           '2020': '2020', 'film': 'film',
                           '428': 'st428', '431': 'st431-2', '432': 'st432-1', 'jedec-p22': 'jedec-p22',
                           'unknown': 'unspec'}
    matrix_f2m_transfer = {'gbr': "0", '709': "1", '470': "5", '601': "2", '170': "6", '240': "7",
                           '2020c': "10", '2020nc': "9", 'fcc': "4", 'ycgco': "8", 'ictcp': "2",
                           'derived-nc': "2", 'derived-c': "2",
                           'unknown': "2", 'reserved': "3"}
    transf_f2m_transfer = {'709': "1", '470m': "2", '470bg': "2", '601': "2", '240': "7",
                           '2020': "14", '2084': "16", 'arib': "18", 'linear': "8",
                           'log': "9",
                           'unknown': "2", 'reserved': "0", '170': "6"}
    primar_f2m_transfer = {'709': "1", '470m': "4", '470bg': "5", '601': "2", '170': "6", '240': "7",
                           '2020': "9", 'film': "2",
                           '428': "10", '431': "2", '432': "2", 'jedec-p22': "22",
                           'unknown': "2", 'reserved': "0"}
    vrange_f2m_transfer = {'unknown': "0", 'tv': "1", 'full': "2"}

    ffmpeg_color_tag_map = {'-color_range': ['color_range', 'tv'],
                            '-color_primaries': ['color_primaries', 'bt709'],
                            '-colorspace': ['color_space', 'bt709'],
                            '-color_trc': ['color_transfer', 'bt709']}
    encc_color_tag_map = {'--colorrange': ['color_range', 'tv'],
                          '--colorprim': ['color_primaries', 'bt709'],
                          '--colormatrix': ['color_space', 'bt709'],
                          '--transfer': ['color_transfer', 'bt709']}
    mkvmerge_color_tag_map = {'--color-range': ['color_range', "1"],
                              '--color-primaries': ['color_primaries', "1"],
                              '--color-matrix-coefficients': ['color_space', "1"],
                              '--color-transfer-characteristics': ['color_transfer', "1"]}

    scale_filter_colorspace = {'bt709': 'bt709', 'fcc': 'fcc', 'bt470bg': 'bt470bg', 'smpte170m': 'smpte170m',
                               'smpte240m': 'smpte240m', 'bt2020': 'bt2020', 'bt2020nc': 'bt2020'}

    @staticmethod
    def is_unknown(tag: str):
        if not len(tag):
            return True
        if 'un' in tag or 'reserved' in tag:
            return True
        return False

    @staticmethod
    def ffmpeg_to_another(value: str, transfer: dict, default):
        """
        transfer ffmpeg's color key in another form
        :param value: ffmpeg's key
        :param transfer: vs key
        :param default: str or int
        :return:
        """
        if 'un' in value:  # unknown
            return default
        for k in transfer:
            if k in value:
                return transfer[k]
        return default

    @staticmethod
    def ffmpeg_to_vspipe_color_triple_tag(color_tag: dict):
        matrix = ColorTag.ffmpeg_to_another(color_tag['color_space'], ColorTag.matrix_f2v_transfer, '709')
        transf = ColorTag.ffmpeg_to_another(color_tag['color_transfer'], ColorTag.transf_f2v_transfer, '709')
        primar = ColorTag.ffmpeg_to_another(color_tag['color_primaries'], ColorTag.primar_f2v_transfer, '709')
        return matrix, transf, primar

    @staticmethod
    def ffmpeg_to_mkvmerge_color_tag(color_tag):
        color_tag['color_range'] = ColorTag.ffmpeg_to_another(color_tag['color_range'], ColorTag.vrange_f2m_transfer, "unknown")
        color_tag['color_space'] = ColorTag.ffmpeg_to_another(color_tag['color_space'], ColorTag.matrix_f2m_transfer, "unknown")
        color_tag['color_transfer'] = ColorTag.ffmpeg_to_another(color_tag['color_transfer'], ColorTag.transf_f2m_transfer, "unknown")
        color_tag['color_primaries'] = ColorTag.ffmpeg_to_another(color_tag['color_primaries'], ColorTag.primar_f2m_transfer, "unknown")
        return color_tag

    @staticmethod
    def get_color_tag_dict(color_tag: dict, color_tag_map: dict):
        """
        Get color tag cli key-value from color tag
        :param color_tag:
        :param color_tag_map:
        :return:
        """
        output_dict = {}

        for ct in color_tag_map:
            ct_data = color_tag[color_tag_map[ct][0]]
            if ColorTag.is_unknown(ct_data):  # FFmpeg Readflow does not support `reserved` tag
                continue  # do not use `default` value, SVFI 3.32.4-alpha
                # ct_data = color_tag_map[ct][1]  # defaultQ
            output_dict.update({ct: str(ct_data)})
        return output_dict

    @staticmethod
    def get_ffmpeg_color_tag_dict(color_tag: dict):
        color_dict = ColorTag.get_color_tag_dict(color_tag, ColorTag.ffmpeg_color_tag_map)
        # bt470 special judge
        if re.search('rgb|gbr|bgr', color_tag['color_space']):
            color_dict.update({'-colorspace': 'bt709',
                               '-color_trc': 'bt709',
                               '-color_primaries': 'bt709'})
        elif '470' in color_tag['color_space']:
            # PAL
            color_dict.update({'-colorspace': 'bt470bg',
                               '-color_trc': 'gamma28',
                               '-color_primaries': 'bt470bg'})
        elif '170' in color_tag['color_space']:
            # NTSC
            color_dict.update({'-colorspace': 'smpte170m',
                               '-color_trc': 'smpte170m',
                               '-color_primaries': 'smpte170m'})
        return color_dict

    @staticmethod
    def get_encc_color_tag_dict(color_tag: dict):
        return ColorTag.get_color_tag_dict(color_tag, ColorTag.encc_color_tag_map)

    @staticmethod
    def get_mkvmerge_color_tag_dict(color_tag: dict):
        return ColorTag.get_color_tag_dict(color_tag, ColorTag.mkvmerge_color_tag_map)


class RGB_TYPE:
    SIZE = 65535.
    DTYPE = np.uint8 if SIZE == 255. else np.uint16

    @staticmethod
    def change_8bit(d8: bool):
        if d8:
            RGB_TYPE.SIZE = 255.
            RGB_TYPE.DTYPE = np.uint8 if RGB_TYPE.SIZE == 255. else np.uint16


class LUTS_TYPE(enum.Enum):
    NONE = 0
    SaturationPQ = 1
    ColorimetricPQ = 2
    ColorimetricHLG = 3
    DV84 = 4

    PQ = 0xfff
    HLG = 0xffe
    SDR = 0xffd

    @staticmethod
    def get_lut_path(lut_type, absolute=False):
        if lut_type is LUTS_TYPE.NONE:
            path = "?.cube"
        elif lut_type is LUTS_TYPE.SaturationPQ:
            path = "1x3d.cube"
        elif lut_type is LUTS_TYPE.ColorimetricPQ:
            path = "1x3d2.cube"
        elif lut_type is LUTS_TYPE.ColorimetricHLG:
            path = "2x3d.cube"
        elif lut_type is LUTS_TYPE.DV84:
            path = "3x3d.cube"
        else:
            path = "?.cube"
        if absolute:
            return os.path.join(appDir, path)
        else:
            return path

    @staticmethod
    def get_lut_colormatrix(lut_type):
        """
        Get colormatrix for lut
        :param lut_type:
        :return:
        """
        if lut_type in [LUTS_TYPE.SaturationPQ, LUTS_TYPE.ColorimetricPQ]:
            return LUTS_TYPE.PQ
        elif lut_type in [LUTS_TYPE.ColorimetricHLG, LUTS_TYPE.DV84]:
            return LUTS_TYPE.HLG
        return LUTS_TYPE.SDR


class ALGO_TYPE(enum.Enum):
    @staticmethod
    def get_model_version(model_path: str):
        raise NotImplementedError()

    @staticmethod
    def get_available_algos_dirs(algo_type: str):
        algo_dir = os.path.join(appDir, "models", algo_type)
        if not os.path.exists(algo_dir):
            return []
        algos = [os.path.join(algo_dir, d) for d in os.listdir(algo_dir)]
        return algos

    @classmethod
    def get_available_algos(cls, algo_type: str, is_full_path=False):
        algos = cls.get_available_algos_dirs(algo_type)
        algos = list(filter(lambda x: not os.path.isfile(x), algos))
        if not is_full_path:
            algos = list(map(lambda x: os.path.basename(x), algos))
        return algos

    @staticmethod
    def get_available_models_dirs(algo_type: str, algo: str):
        model_dir = os.path.join(appDir, "models", algo_type, algo, "models")
        models = [os.path.join(model_dir, d) for d in os.listdir(model_dir)]
        return models

    @classmethod
    def get_available_models(cls, algo_type: str, algo: str, is_file=False, is_full_path=False, ext: list = None,
                             stop_words: list = None):
        models = cls.get_available_models_dirs(algo_type, algo)
        if is_file:
            models = list(filter(lambda x: os.path.isfile(x), models))
            if ext is not None:
                models = list(filter(lambda x: os.path.splitext(x)[1] in ext, models))
        else:
            models = list(filter(lambda x: not os.path.isfile(x), models))
        if not is_full_path:
            models = list(map(lambda x: os.path.basename(x), models))
        if stop_words is not None:
            models = list(filter(lambda x: not re.search("|".join(stop_words), x), models))
        return models


class GLOBAL_PARAMETERS:
    CURRENT_CUDA_ID = 0
    FLOW_ABS_MAX = 0
    LANGUAGE = 0  # 0: CHN, 1: OTHER LANG
    RESAMPLE_VFI_LOCK = threading.Lock()


class VFI_TYPE(ALGO_TYPE):
    """

    """
    RIFEv2       =          0b10000000000000000000
    RIFEv3       =          0b01000000000000000000
    RIFEv4       =          0b00100000000000000000
    RIFEvPlus    =          0b00010000000000000000
    IFRNET       =          0b00001000000000000000
    IFUNET       =          0b00000100000000000000
    NCNNv2       =          0b00000010000000000000
    NCNNv3       =          0b00000001000000000000
    NCNNv4       =          0b00000000100000000000
    NCNN_DAIN    =          0b00000000010000000000
    IFNeXt       =          0b00000000001000000000
    GmfSs        =          0b00000000000100000000
    GmfSs_xd     =          0b00000000000010000000
    Umss         =          0b00000000000001000000
    Vfss         =          0b00000000000000100000
    EMA          =          0b00000000000000010000
    NCNNv4S      =          0b00000000000000001000  # From Styler0Dollar
    RIFEv4DIO    =          0b00000000000000000100  # Fusion Model based on Multi-Scale of RIFE v4.6 (official 4.6), 'dio' for 'dionysus'
    TensorRT     =          0b00000000000000000010  # NOQA: VFI Flow not passed
    AUTO         =          0b00000000000000000001

    DS =                    0b11110000000000000001
    TTA =                   0b11000000100000010001
    ENSEMBLE =              0b11110000100110000001
    MULTICARD =             0b11111100001111010101  # dilapidated
    OUTPUTMODE =            0b00010000000000000001
    FP16 =                  0b11111100001111110011

    @staticmethod
    def get_model_version(model_path: str):
        """
        Return Model Type of VFI
        :param model_path:
        :return:

        """
        model_path = model_path.lower()
        # AUTO
        if 'auto' in model_path:
            current_model_index = VFI_TYPE.AUTO
        elif 'tensorrt' in model_path or 'onnx' in model_path:  # TODO: check use onnx -> trt only
            current_model_index = VFI_TYPE.TensorRT
        # CUDA Model
        elif 'rpa_' in model_path or 'rpr_' in model_path:  # RIFE Plus Anime, prior than anime
            current_model_index = VFI_TYPE.RIFEvPlus  # RIFEv New from Master Zhe
        elif 'anime_' in model_path:
            current_model_index = VFI_TYPE.RIFEv2  # RIFEv2
        elif 'official_' in model_path:
            if '2.' in model_path:
                current_model_index = VFI_TYPE.RIFEv2  # RIFEv2.x
            elif '3.' in model_path:
                current_model_index = VFI_TYPE.RIFEv3
            else:  # if '4.' in model_path:
                current_model_index = VFI_TYPE.RIFEv4
        elif '_dionysus' in model_path:
            current_model_index = VFI_TYPE.RIFEv4DIO
        elif 'ifrnet' in model_path:
            current_model_index = VFI_TYPE.IFRNET
        elif 'ifunet' in model_path:
            current_model_index = VFI_TYPE.IFUNET
        elif 'ifnext' in model_path:
            current_model_index = VFI_TYPE.IFNeXt
        elif 'gmfss_xd' in model_path:
            current_model_index = VFI_TYPE.GmfSs_xd
        elif 'gmfss' in model_path:
            current_model_index = VFI_TYPE.GmfSs
        elif 'umss' in model_path:
            current_model_index = VFI_TYPE.Umss
        elif 'vfss' in model_path:
            current_model_index = VFI_TYPE.Vfss
        # NCNN Model:
        elif 'rife-v2' in model_path:
            current_model_index = VFI_TYPE.NCNNv2  # RIFEv2.x
        elif 'rife-v3' in model_path:
            current_model_index = VFI_TYPE.NCNNv3  # RIFEv3.x
        elif 'rife-v4' in model_path:
            current_model_index = VFI_TYPE.NCNNv4  # RIFEv4.x
            if '_ensemble' in model_path:
                current_model_index = VFI_TYPE.NCNNv4S  # Model From Styler0Dollar
        elif 'dain' in model_path:
            current_model_index = VFI_TYPE.NCNN_DAIN
        elif 'ema' in model_path:
            current_model_index = VFI_TYPE.EMA
        # Default
        else:
            current_model_index = VFI_TYPE.RIFEv2  # default RIFEv2
        return current_model_index

    @staticmethod
    def update_current_gpu_id(gpu_id: int):
        GLOBAL_PARAMETERS.CURRENT_CUDA_ID = gpu_id


VFI_ANYTIME = [VFI_TYPE.RIFEv4, VFI_TYPE.RIFEvPlus, VFI_TYPE.RIFEv4DIO,
               VFI_TYPE.IFUNET, VFI_TYPE.IFNeXt, VFI_TYPE.GmfSs, VFI_TYPE.GmfSs_xd, VFI_TYPE.Umss, VFI_TYPE.Vfss,
               VFI_TYPE.IFRNET, VFI_TYPE.EMA,
               VFI_TYPE.AUTO]
VFI_NCNN = [VFI_TYPE.NCNNv2, VFI_TYPE.NCNNv3, VFI_TYPE.NCNNv4, VFI_TYPE.NCNNv4S, VFI_TYPE.NCNN_DAIN]
VFI_ANYTIME.extend(VFI_NCNN)
VFI_FILE_MODELS = [VFI_TYPE.TensorRT, VFI_TYPE.EMA]
VFI_MULTI_INPUT = [VFI_TYPE.Vfss, VFI_TYPE.GmfSs_xd]


class SR_TYPE(ALGO_TYPE):
    """
    Model Type Index Table of Super Resolution used by SVFI
    """
    Anime4K       = 0b1000000000000000
    RealESR       = 0b0100000000000000
    RealCUGAN     = 0b0010000000000000
    NcnnCUGAN     = 0b0001000000000000
    WaifuCUDA     = 0b0000100000000000
    Waifu2x       = 0b0000010000000000
    NcnnRealESR   = 0b0000001000000000
    BasicVSRPP    = 0b0000000100000000
    RealBasicVSR  = 0b0000000010000000
    BasicVSRPPR   = 0b0000000001000000
    PureBasicVSR  = 0b0000000000100000
    TensorRT      = 0b0000000000010000
    FTVSR         = 0b0000000000001000
    AnimeSR       = 0b0000000000000100
    NvidiaSR      = 0b0000000000000010
    AUTO          = 0b0000000000000001

    @staticmethod
    def get_available_algos_dirs(algo_type: str):
        algo_dir = os.path.join(appDir, "models", algo_type)
        algos = [os.path.join(algo_dir, d) for d in os.listdir(algo_dir)]
        # algo_dir = os.path.join(appDir, "vspipe/vapoursynth64/coreplugins/models")
        # algos.extend([os.path.join(algo_dir, d) for d in ['cugan_trt', 'RealESRGANv2_trt']])
        return algos

    @staticmethod
    def get_available_models_dirs(algo_type: str, algo: str):
        model_dir = os.path.join(appDir, "models", algo_type, algo, "models")
        if os.path.exists(model_dir):
            models = [os.path.join(model_dir, d) for d in os.listdir(model_dir)]
        else:
            models = []
        # model_dir = os.path.join(appDir, "vspipe/vapoursynth64/coreplugins/models", algo)
        # if os.path.exists(model_dir):
        #     models.extend([os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.splitext(d)[1] == '.onnx'])
        return models

    @staticmethod
    def get_model_version(model_path: str):
        """
        Return Model Type of SR
        :param model_path:
        :return:

        Register Procedure already updated to Routines of Maintenance
        """
        model_path = model_path.lower()  # algo name (dir path) or model name
        current_model_index = SR_TYPE.AUTO

        # AUTO
        if 'auto' in model_path:
            current_model_index = SR_TYPE.AUTO
        # ONNX/TRT model
        # elif 'onnx' in model_path or '_trt' in model_path:  # use onnx -> trt only
        elif 'tensorrt' in model_path or 'onnx' in model_path:  # TODO: check use onnx -> trt only
            current_model_index = SR_TYPE.TensorRT
        # Render Model
        elif 'anime4k' in model_path:
            current_model_index = SR_TYPE.Anime4K
        # CUDA Model
        elif 'realcugan' in model_path:
            current_model_index = SR_TYPE.RealCUGAN
        elif 'ncnnrealesr' in model_path:
            current_model_index = SR_TYPE.NcnnRealESR
        elif 'realesr' in model_path:
            current_model_index = SR_TYPE.RealESR
        elif 'waifucuda' in model_path:
            current_model_index = SR_TYPE.WaifuCUDA
        # NCNN Model
        elif 'ncnncugan' in model_path:
            current_model_index = SR_TYPE.NcnnCUGAN
        elif 'waifu2x' in model_path:
            current_model_index = SR_TYPE.Waifu2x
        elif 'basicvsrplusplusrestore' in model_path:
            current_model_index = SR_TYPE.BasicVSRPPR
        elif 'basicvsrplusplus' in model_path:
            current_model_index = SR_TYPE.BasicVSRPP
        elif 'ftvsr' in model_path:
            current_model_index = SR_TYPE.FTVSR
        elif 'realbasicvsr' in model_path:
            current_model_index = SR_TYPE.RealBasicVSR
        elif 'purebasicvsr' in model_path:
            current_model_index = SR_TYPE.PureBasicVSR
        elif 'animesr' in model_path:
            current_model_index = SR_TYPE.AnimeSR
        elif 'nvidiasr' in model_path:
            current_model_index = SR_TYPE.NvidiaSR
        return current_model_index


SR_NCNN = [SR_TYPE.NcnnCUGAN, SR_TYPE.Waifu2x, SR_TYPE.Anime4K, SR_TYPE.NcnnRealESR]
SR_MULTIPLE_INPUTS = [SR_TYPE.FTVSR, SR_TYPE.BasicVSRPP, SR_TYPE.RealBasicVSR, SR_TYPE.BasicVSRPPR,
                      SR_TYPE.PureBasicVSR]
SR_FILE_MODELS = [SR_TYPE.RealESR, SR_TYPE.WaifuCUDA, SR_TYPE.RealCUGAN, SR_TYPE.Anime4K, SR_TYPE.PureBasicVSR,
                  SR_TYPE.TensorRT, SR_TYPE.AnimeSR, SR_TYPE.NvidiaSR]


# class P5X_STATE(enum.Enum):
#     INACTIVE = -1
#     POST = 1
#     BEFORE = 0


class DEDUP_MODE(enum.Enum):
    NONE = 0  # no dedup
    TRUMOTION = 1  # TruMotion
    SINGLE = 2  # single threshold
    D2 = 3  # (traditional) dedup shot on twos
    D3 = 4  # (traditional) dedup shot on twos/threes
    RESAMPLE = 5  # downsample and interpolate
    DIFF = 6  # first order dedup
    RECON = 7  # dual referenced reconstruction
    DIST = 0xffff  # frame distribution


USER_PRIVACY_GRANT_PATH = os.path.join(appDir, "PrivacyStat.md")
