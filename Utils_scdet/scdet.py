import os
import traceback
from collections import deque
from queue import Queue

import cv2
import numpy as np
from sklearn import linear_model


class TransitionDetectionBase:
    def __init__(self, scene_queue_length, scdet_threshold=12, pure_scene_threshold=10, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=80):
        """
            :param scene_queue_length: 用于转场识别的帧队列长
            :param scdet_threshold: 非固定转场识别模式下的阈值
            :param no_scdet: 无转场检测
            :param use_fixed_scdet: 使用固定转场识别
            :param fixed_max_scdet: 所有转场识别模式下的死值；固定转场识别模式下的阈值
            """
        self.scdet_threshold = scdet_threshold
        self.pure_scene_threshold = pure_scene_threshold
        self.use_fixed_scdet = use_fixed_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.scene_checked_queue = deque(maxlen=self.scene_stack_len // 2)  # 已判断的转场absdiff特判队列
        self.dead_thres = fixed_max_scdet
        self.born_thres = 2  # 判定为转场的最小阈值
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.no_scdet = no_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}

        self.norm_resize = (300, 300)

    def get_diff(self, img0, img1):
        """
        method of getting diff between img0 and img1, could be overwritten by subclass so as to use different diff
        :param img0:
        :param img1:
        :return:
        """
        return self._get_norm_img_diff(img0, img1)

    def _get_u1_from_u2_img(self, img):
        if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
            try:
                img = img.view(np.uint8)[:, :, ::2]  # default to uint8
            except ValueError:
                img = np.ascontiguousarray(img, dtype=np.uint16).view(np.uint8)[:, :, ::2]  # default to uint8
        return img

    def _get_norm_img(self, img):
        img = self._get_u1_from_u2_img(img)
        if img.shape[0] > 1000:
            img = img[::4, ::4, 0]
        else:
            img = img[::2, ::2, 0]
        img = cv2.resize(img, self.norm_resize)
        img = cv2.equalizeHist(img)  # 进行直方图均衡化
        return img

    def _check_pure_img(self, img):
        try:
            if np.var(img[::4, ::4, 0]) < self.pure_scene_threshold:
                return True
            return False
        except:
            return False

    def _get_norm_img_diff(self, img0, img1):
        if np.allclose(img0[::4, ::4, 0], img1[::4, ::4, 0]):
            return 0
        img0 = self._get_norm_img(img0)
        img1 = self._get_norm_img(img1)
        diff = cv2.absdiff(img0, img1).mean()
        return diff

    def _check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def _check_var(self):
        coef, intercept = self._check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return sub_array.var() ** 0.65

    def _judge_mean(self, diff):
        var_before = self._check_var()
        self.absdiff_queue.append(diff)
        var_after = self._check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            # Detect new scene
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}")
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def save_scene(self, title: str):
        raise NotImplementedError()

    def check_scene(self, _img1, _img2, add_diff=False, no_diff=False, use_diff=-1, **kwargs) -> bool:
        """
        Check if current scene is scene
        :param use_diff:
        :param _img2:
        :param _img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回真
        """

        if self.no_scdet:
            return False

        img1 = _img1.copy()
        img2 = _img2.copy()
        self.img1 = img1
        self.img2 = img2

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.get_diff(self.img1, self.img2)

        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            else:
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
                return True

        # 检测开头黑场
        if diff < 0.001:
            # 000000
            if self._check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif len(self.black_scene_queue) and np.mean(self.black_scene_queue) == 0:
            # 检测到00000001
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        # Check really hard scene at the beginning
        if diff > self.dead_thres:
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        # Duplicate Frames Special Judge
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        # Judge
        return self._judge_mean(diff)

    def update_scene_status(self, recent_scene: int, scene_type: str):
        # 更新转场检测状态
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class TransitionDetectionHSV(TransitionDetectionBase):
    hsl_weight = [1., 1., 1.]

    @staticmethod
    def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
        """Return the mean average distance in pixel values between `left` and `right`.
        Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
        """
        num_pixels: float = float(left.shape[0] * left.shape[1])
        return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

    def get_img_hsv(self, img):
        img = self._get_u1_from_u2_img(img)
        if img.shape[0] > 1000:
            img = img[::4, ::4, :]
        else:
            img = img[::2, ::2, :]
        hue, sat, lum = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return hue, sat, lum

    def get_diff(self, img0, img1):
        hsl0, hsl1 = self.get_img_hsv(img0), self.get_img_hsv(img1)

        score_components = [self.mean_pixel_distance(hsl0[i], hsl1[i]) for i in range(3)]

        diff: float = (
                sum(component * weight for (component, weight) in zip(score_components, self.hsl_weight))
                / sum(abs(weight) for weight in self.hsl_weight))
        return diff


class SvfiTransitionDetection(TransitionDetectionHSV):
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=16, pure_scene_threshold=10,
                 no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=60, scdet_output=False):
        """

        :param project_dir: 项目所在文件夹
        :param scdet_output: 是否输出转场
        """
        super().__init__(scene_queue_length=scene_queue_length,
                         scdet_threshold=scdet_threshold, pure_scene_threshold=pure_scene_threshold,
                         no_scdet=no_scdet, use_fixed_scdet=use_fixed_scdet, fixed_max_scdet=fixed_max_scdet)
        from Utils_scdet.utils import Tools
        from Utils_scdet.StaticParameters import RGB_TYPE
        self.utils = Tools  # static
        self.scene_dir = os.path.join(project_dir, "scene")  # save dir path
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scene_stack = Queue(maxsize=scene_queue_length)
        self.scdet_output = scdet_output
        self.rgb_size = RGB_TYPE.SIZE

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])),
                                    interpolation=cv2.INTER_AREA)
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(self.rgb_size), 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
        except Exception:
            traceback.print_exc()


class VsTransitionDetection(TransitionDetectionHSV):
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=16, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=60, scdet_output=False):
        super().__init__(scene_queue_length=scene_queue_length, scdet_threshold=scdet_threshold,
                         no_scdet=no_scdet, use_fixed_scdet=use_fixed_scdet, fixed_max_scdet=fixed_max_scdet)
        self.scene_dir = os.path.join(project_dir, "scene")  # save dir path
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scdet_output = scdet_output

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            # print(comp_stack.shape, file=sys.stderr)
            # print(self.img1.shape, file=sys.stderr)
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])),
                                    interpolation=cv2.INTER_AREA)
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
        except Exception:
            traceback.print_exc()

