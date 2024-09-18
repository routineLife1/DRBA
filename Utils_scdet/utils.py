# coding: utf-8
import datetime
import glob
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import traceback
from collections import deque
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np
import psutil
from sklearn import linear_model

from Utils_scdet.StaticParameters import RGB_TYPE, IS_RELEASE, IS_CLI, appDir, SupportFormat
# from skvideo.utils import startupinfo


class DefaultConfigParser(ConfigParser):
    """
    自定义参数提取
    """

    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value


class CliFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Tools:
    resize_param = (300, 300)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        pass

    @staticmethod
    def fillQuotation(_str):
        if _str[0] != '"':
            return f'"{_str}"'
        else:
            return _str

    @staticmethod
    def get_logger(name, log_path, debug=False, silent=False, is_subprocess=False):
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if IS_CLI:
            logger_formatter = CliFormatter()
        else:
            logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
            if IS_RELEASE:
                logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(levelname)s - %(message)s')
            if is_subprocess:
                logger_formatter = logging.Formatter(f'SUB - %(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')

        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{name}-{datetime.datetime.now().date()}.log")

        txt_handler = logging.FileHandler(logger_path, encoding='utf-8')

        txt_handler.setFormatter(logger_formatter)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logger_formatter)

        if not silent:
            logger.addHandler(console_handler)
            logger.addHandler(txt_handler)
        else:
            logger.handlers.clear()
        return logger

    @staticmethod
    def make_dirs(dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    @staticmethod
    def gen_next(gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    @staticmethod
    def dict2Args(d: dict):
        args = []
        for key in d.keys():
            args.append(key)
            if len(d[key]):
                args.append(d[key])
        return args

    @staticmethod
    def clean_parsed_config(args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                # print(f"INFO: Find Empty Arguments at '{a}'", file=sys.stderr)
                args[a] = ""
        return args

    @staticmethod
    def check_pure_img(img1):
        try:
            if np.var(img1[::4, ::4, 0]) < 10:
                return True
            return False
        except:
            return False

    @staticmethod
    def check_non_ascii(s: str):
        ascii_set = set(string.printable)
        _s = ''.join(filter(lambda x: x in ascii_set, s))
        if s != _s:
            return True
        else:
            return False

    @staticmethod
    def get_u1_from_u2_img(img: np.ndarray):
        if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
            try:
                img = img.view(np.uint8)[:, :, ::2]  # default to uint8
            except ValueError:
                img = np.ascontiguousarray(img, dtype=np.uint16).view(np.uint8)[:, :, ::2]  # default to uint8
        return img

    @staticmethod
    def get_norm_img(img1, resize=True):
        img1 = Tools.get_u1_from_u2_img(img1)
        if img1.shape[0] > 1000:
            img1 = img1[::4, ::4, 0]
        else:
            img1 = img1[::2, ::2, 0]
        if resize and img1.shape[0] > Tools.resize_param[0]:
            img1 = cv2.resize(img1, Tools.resize_param)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        return img1

    @staticmethod
    def get_norm_img_diff(img1, img2, resize=True, is_flow=False) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :param is_flow: bool
        :return: float
        """

        def fd(_i0, _i1):
            """
            Calculate Flow Distance
            :param _i0: np.ndarray
            :param _i1: np.ndarray
            :return:
            """
            prev_gray = cv2.cvtColor(_i0, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(_i1, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow=None,
                                                pyr_scale=0.5, levels=1, winsize=64, iterations=20,
                                                poly_n=5, poly_sigma=1.1, flags=0)
            x = flow[:, :, 0]
            y = flow[:, :, 1]
            return np.linalg.norm(x) + np.linalg.norm(y)

        if np.array_equal(img1[::4, ::4, 0], img2[::4, ::4, 0]):
            return 0

        if is_flow:
            img1 = Tools.get_u1_from_u2_img(img1)
            img2 = Tools.get_u1_from_u2_img(img2)
            i0 = cv2.resize(img1, (64, 64))
            i1 = cv2.resize(img2, (64, 64))
            diff = fd(i0, i1)
        else:
            img1 = Tools.get_norm_img(img1, resize)
            img2 = Tools.get_norm_img(img2, resize)
            # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            diff = cv2.absdiff(img1, img2).mean()

        return diff

    @staticmethod
    def get_norm_img_flow(img1, img2, resize=True, flow_thres=1) -> (int, np.array):
        """
        Normalize Difference
        :param flow_thres: 光流移动像素长
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return:  (int, np.array)
        """
        prevgray = Tools.get_norm_img(img1, resize)
        gray = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 绘制线
        step = 10
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []
        flow_cnt = 0

        for l in lines:
            if math.sqrt(math.pow(l[0][0] - l[1][0], 2) + math.pow(l[0][1] - l[1][1], 2)) > flow_thres:
                flow_cnt += 1
                line.append(l)

        cv2.polylines(prevgray, line, 0, (0, 255, 255))
        comp_stack = np.hstack((prevgray, gray))
        return flow_cnt, comp_stack

    @staticmethod
    def get_filename(path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def get_mixed_scenes(img0, img1, n):
        """
        return n-1 images
        :param img0:
        :param img1:
        :param n:
        :return:
        """
        step = 1 / n
        beta = 0
        output = list()

        def normalize_img(img):
            if img.dtype in (np.dtype('>u2'), np.dtype('<u2')):
                img = img.astype(np.uint16)
            return img

        img0 = normalize_img(img0)
        img1 = normalize_img(img1)
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output

    @staticmethod
    def get_fps(path: str):
        """
        Get Fps from path
        :param path:
        :return: fps float
        """
        if not os.path.isfile(path):
            return 0
        try:
            if not os.path.isfile(path):
                input_fps = 0
            else:
                input_stream = cv2.VideoCapture(path)
                input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            return input_fps
        except Exception:
            return 0

    @staticmethod
    def get_existed_chunks(project_dir: str):
        chunk_paths = []
        for chunk_p in os.listdir(project_dir):
            if re.match("chunk-\d+-\d+-\d+\.\w+", chunk_p):
                chunk_paths.append(chunk_p)

        if not len(chunk_paths):
            return chunk_paths, -1, -1

        chunk_paths.sort()
        last_chunk = chunk_paths[-1]
        chunk_cnt, last_frame = re.findall('chunk-(\d+)-\d+-(\d+).*?', last_chunk)[0]
        return chunk_paths, int(chunk_cnt), int(last_frame)

    @staticmethod
    def get_available_images(folder: str):
        img_list = []
        for ext in SupportFormat.img_inputs:
            glob_expression = glob.escape(folder) + f"/*{ext}"
            img_list.extend(glob.glob(glob_expression))
        return img_list

    @staticmethod
    def get_custom_cli_params(_command: str):
        command_params = _command.split('||')
        command_dict = dict()
        param = ""
        for command in command_params:
            command = command.strip().replace("\\'", "'").replace('\\"', '"').strip('\\')
            if command.startswith("-"):
                if param != "":
                    command_dict.update({param: ""})
                param = command
            else:
                command_dict.update({param: command})
                param = ""
        if param != "":  # final note
            command_dict.update({param: ""})
        return command_dict

    @staticmethod
    def popen(args, is_stdout=False, is_stderr=False, *pargs, **kwargs):
        """
        Used to fetch result from stderr or stdout that needs manual process control
        :param args: list of strs or str
        :param is_stdout:
        :param is_stderr:
        :param pargs:
        :param kwargs:
        :return:
        """
        p = subprocess.Popen(args, startupinfo=startupinfo,
                             stdout=subprocess.PIPE if is_stdout else None,
                             stderr=subprocess.PIPE if is_stderr else None,
                             encoding='utf-8', *pargs, **kwargs)
        return p

    @staticmethod
    def md5(d: str):
        m = hashlib.md5(d.encode(encoding='utf-8'))
        return m.hexdigest()

    @staticmethod
    def get_pids():
        """
        get key-value of pids
        :return: dict {pid: pid-name}
        """
        pid_dict = {}
        pids = psutil.pids()
        for pid in pids:
            try:
                p = psutil.Process(pid)
                pid_dict[pid] = p.name()
            except psutil.NoSuchProcess:
                pass
            # print("pid-%d,pname-%s" %(pid,p.name()))
        return pid_dict

    @staticmethod
    def kill_svfi_related(pid: int, is_rude=False, is_kill_ols=False):
        """

        :param is_kill_ols:
        :param is_rude:
        :param pid: PID of One Line Shot Args.exe
        :return:
        """

        try:
            p = Tools.popen(f'wmic process where parentprocessid={pid} get processid', is_stdout=True)
            related_pids = p.stdout.readlines()
            p.stdout.close()
            related_pids = [i.strip() for i in related_pids if i.strip().isdigit()]
            if is_kill_ols:
                related_pids.append(str(pid))
            if not len(related_pids):
                return
            taskkill_cmd = "taskkill "
            for p in related_pids:
                taskkill_cmd += f"/pid {p} "
            taskkill_cmd += "/f"
            try:
                p = Tools.popen(taskkill_cmd)
                p.wait(timeout=15)
            except:
                pass

            must_kill = ['trtexec.exe', 'VSPipe.exe']
            taskkill_cmd = "taskkill "
            for im in must_kill:
                taskkill_cmd += f"/im {im} "
            taskkill_cmd += "/f"
            try:
                p = Tools.popen(taskkill_cmd)
                p.wait(timeout=15)
            except:
                pass
            # raise OSError("Test")
        except FileNotFoundError:
            if is_rude:
                pids = Tools.get_pids()
                for pid, pname in pids.items():
                    if any([i in pname for i in ['ffmpeg', 'ffprobe', 'one_line_shot_args', 'QSVEncC64', 'NVEncC64',
                                                 'SvtHevcEncApp', 'SvtVp9EncApp', 'SvtAv1EncApp', 'vspipe',
                                                 'trtexec']]):
                        try:
                            os.kill(pid, signal.SIGABRT)
                        except Exception as e:
                            traceback.print_exc()
                        print(f"Warning: Kill Process before exit: {pname}", file=sys.stderr)
                return
            pass

    @staticmethod
    def get_plural(i: int):
        if i > 0:
            if i % 2 != 0:
                return i + 1
        return i

    @staticmethod
    def is_oom() -> bool:
        mem = psutil.virtual_memory()
        return True if mem.percent > 85.0 else False

    @staticmethod
    def is_valid_filename(filename: str) -> bool:
        if any([i in filename for i in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']]):
            return False
        return True

    @staticmethod
    def text_img(img, text):
        img = cv2.putText(np.ascontiguousarray(img), text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(RGB_TYPE.SIZE), 0, 0), 2)
        return img


class AnytimeFpsIndexer:
    def __init__(self, input_fps: float, output_fps: float, scene_list: list = None):
        """
        Case 1:
        24 -> 60, ratio = 0.4

        0        1           2
        |    \   |   \   \   |
        0.0 0.4 0.8 1.2 1.6 2.0

        Duplicate Frames using input seq as ref

        Case 2:
        72 -> 60: ratio = 1.2

        0    1   2   3   4   5
        |    |   |       |   |
        0.0 1.2 2.4     3.6 4.8 6.0

        Dropping Frames using input seq as ref
        TODO: high fps --map--> low still results VA out of sync

        :param input_fps:
        :param output_fps:
        :param scene_list:
        """
        self.inputfps = input_fps
        self.outputfps = output_fps
        self.ratio = self.inputfps / self.outputfps
        self.is_reversed = self.ratio > 1
        assert not self.is_reversed, AssertionError("Currently does not support fps mapping from high to low ")
        self.iNow = 0
        self.oNow = 0
        if scene_list is None:
            scene_list = []
        self.scene_list = scene_list
        self.is_current_scene = False

    def isCurrentScene(self):
        return self.is_current_scene

    def isReversed(self):
        return self.is_reversed

    def isCurrentDup(self):
        """
        This method will update its self-status when called

        SPECIAL USAGE: when i/o fps are reversed, isDup stands for frame number that shouldn't be processed, which is used in PipeWriter

        :return:
        """
        iNext = self.iNow + 1

        # Check whether is manual scene first
        if self.iNow in self.scene_list:
            self.is_current_scene = True
        else:
            self.is_current_scene = False

        if abs(self.oNow - self.iNow) <= abs(self.oNow - iNext):
            isDup = True
            # Case 1: the next output frame is Duplicate
            # Case 2: the next output frame is not dropped
        else:
            self.iNow += 1
            isDup = False
            # Case 1: the next output frame is not duplicate, should get new frame
            # Case 2: the next output frame should be dropped
        if self.is_reversed:
            self.iNow += 1
        self.oNow += self.ratio
        return isDup

    def getNow(self):
        return self.iNow


def get_global_settings_from_local_jsons() -> dict:
    path = os.path.join(appDir, "global_advanced_settings.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    return settings


def wrap_to_json(data: dict) -> str:
    data_str = json.dumps(data)
    return data_str


def is_cuda_ext_ok() -> bool:
    return True
    version_file = os.path.join(appDir, "torch", "version.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding='utf-8') as r:
            content = r.read()
            result = re.findall("__version__ = '(\d+)\.(\d+)\.(\d+)", content)
            if not len(result):
                return False
            print(result)
            a, b, c = result[0]
            if int(b) < 9:
                return False
            else:
                return True
    else:
        return False


def is_vspipe_ok() -> bool:
    return os.path.exists(os.path.join(appDir, "vspipe"))


def clean_torch_module():
    """
    Due to SteamPipe bugs,
    distributed torch(1.13.1) package contains the following modules that need to be cleaned"""
    torch_lib_root = os.path.join(appDir, "torch/lib")
    del_torch_lib = ['caffe2_observers.lib',
                     'caffe2_observers.lib',
                     'caffe2_detectron_ops_gpu.dll',
                     'caffe2_detectron_ops_gpu.lib',
                     'caffe2_module_test_dynamic.dll',
                     'caffe2_module_test_dynamic.lib',
                     'caffe2_nvrtc.dll',
                     'caffe2_nvrtc.lib',
                     'caffe2_observers.dll']
    for lib in del_torch_lib:
        path = os.path.join(torch_lib_root, lib)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except:
            pass

    # link pytorch dlls for vsmlrt-cuda
    # they are in torch/lib, link them to vspipe/vapoursynth64/coreplugins/vsmlrt-cuda
    torch_lib_root = os.path.join(appDir, "torch/lib")
    vsmlrt_cuda_root = os.path.join(appDir, "vspipe/vapoursynth64/coreplugins/vsmlrt-cuda")
    dlls = ['cudnn_ops_infer64_8.dll',
            'cudnn_cnn_infer64_8.dll',
            'cudnn64_8.dll',
            'cublasLt64_11.dll',
            'cublas64_11.dll',
            ]
    for dll in dlls:
        src = os.path.join(torch_lib_root, dll)
        dst = os.path.join(vsmlrt_cuda_root, dll)
        if os.path.isfile(src):
            try:
                os.remove(dst)
            except:
                pass
            try:
                os.link(src, dst)
            except:
                pass


if __name__ == "__main__":
    _afi = AnytimeFpsIndexer(96, 24)
    _dup_cnt = 0
    for _i in range(72):
        _iNow, _oNow = _afi.iNow, _afi.oNow
        _is_dup = _afi.isCurrentDup()
        print(f"i = {_iNow}, o = {_oNow}, keep = {_is_dup}")
        _dup_cnt += 1 if _is_dup else 0
    pass
