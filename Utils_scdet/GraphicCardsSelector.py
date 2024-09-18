# coding=utf-8
import traceback
import re
from collections import OrderedDict
import threading

from Utils.StaticParameters import IS_DARWIN, appDir
from Utils.utils import Tools


class GraphicCardsSelector:
    def __init__(self, logger=None):
        self.logger = Tools.get_logger('GraphicCardsSelector', appDir)
        self.cuda_gpu_infos = {'card_cnt': 0, 'card_infos': OrderedDict()}
        self.ffmpeg_gpu_infos = {'card_cnt': 0, 'card_infos': OrderedDict()}
        self.vulkan_gpu_infos = {'card_cnt': 0, 'card_infos': OrderedDict()}

        self._cuda_card_cnt = 0
        self._cuda_card_infos = OrderedDict()
        self._amd_card_cnt = 0
        self._amd_card_infos = OrderedDict()
        self._intel_card_cnt = 0
        self._intel_card_infos = OrderedDict()
        self._vulkan_card_cnt = 0
        self._vulkan_card_infos = OrderedDict()
        self._ffmpeg_card_cnt = 0
        self._ffmpeg_card_infos = OrderedDict()

        self._get_cuda_gpu_infos_thread = threading.Thread(target=self._get_cuda_gpus)
        self._get_ffmpeg_gpu_infos_thread = threading.Thread(target=self._get_ffmpeg_gpus)
        self._get_vulkan_gpu_infos_thread = threading.Thread(target=self._get_vulkan_gpus)
        self._get_cuda_gpu_infos_thread.start()
        self._get_ffmpeg_gpu_infos_thread.start()
        self._get_vulkan_gpu_infos_thread.start()

    @property
    def cuda_card_cnt(self):
        self._get_cuda_gpu_infos_thread.join()
        return self._cuda_card_cnt

    @property
    def cuda_card_infos(self):
        self._get_cuda_gpu_infos_thread.join()
        return self._cuda_card_infos

    @property
    def amd_card_cnt(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._amd_card_cnt

    @property
    def amd_card_infos(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._amd_card_infos

    @property
    def vulkan_card_cnt(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._vulkan_card_cnt

    @property
    def vulkan_card_infos(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._vulkan_card_infos

    @property
    def intel_card_cnt(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._intel_card_cnt

    @property
    def intel_card_infos(self):
        self._get_vulkan_gpu_infos_thread.join()
        return self._intel_card_infos

    @property
    def ffmpeg_card_cnt(self):
        self._get_ffmpeg_gpu_infos_thread.join()
        return self._ffmpeg_card_cnt

    @property
    def ffmpeg_card_infos(self):
        self._get_ffmpeg_gpu_infos_thread.join()
        return self._ffmpeg_card_infos

    def get_cuda_gpus(self) -> (int, str):
        self._get_cuda_gpu_infos_thread.join()
        return self.cuda_gpu_infos['card_cnt'], self.cuda_gpu_infos['card_infos']

    def get_ffmpeg_gpus(self) -> (int, str):
        self._get_ffmpeg_gpu_infos_thread.join()
        return self.ffmpeg_gpu_infos['card_cnt'], self.ffmpeg_gpu_infos['card_infos']

    def get_vulkan_gpus(self) -> (int, str):
        self._get_vulkan_gpu_infos_thread.join()
        return self.vulkan_gpu_infos['card_cnt'], self.vulkan_gpu_infos['card_infos']

    def _get_cuda_gpus(self):
        card_infos = OrderedDict()
        card_cnt = 0
        if IS_DARWIN:
            return 0, {"0": "Apple"}
        try:
            # raise Exception("No NVIDIA Card TEST")  # debug
            import torch
            card_cnt = torch.cuda.device_count()
            for i in range(card_cnt):
                card = torch.cuda.get_device_properties(i)
                info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
                card_infos[f"{i}"] = info
            self.logger.debug(f"CUDA: {card_infos}")
        except:
            self.logger.error("Failed to load status of CUDA Graphic Cards")
            self.logger.error(traceback.format_exc())
        self.cuda_gpu_infos['card_cnt'] = card_cnt
        self.cuda_gpu_infos['card_infos'] = card_infos

        self._cuda_card_cnt = card_cnt
        self._cuda_card_infos = card_infos

    def _get_ffmpeg_gpus(self):
        card_infos = OrderedDict()
        card_cnt = 0
        if IS_DARWIN:
            return 0, {"0": "Apple"}
        try:
            p = Tools.popen("ffmpeg -hide_banner -init_hw_device vulkan -v verbose", is_stderr=True, shell=True)
            r = p.stderr.read()
            p.stderr.close()
            r1 = re.split("\[.*?\] GPU listing:\n", r)[1]
            r2 = re.split("\[.*?\] Device \d", r1)[0]
            card_infos_list = re.findall("\[.*?\]\s+(\d+): (.*?)\n", r2)
            card_cnt = len(card_infos_list)
            for i in range(card_cnt):
                card_infos[card_infos_list[i][0]] = card_infos_list[i][1]
            self.logger.debug(f"FFmpeg: {card_infos}")
        except:
            self.logger.error("Failed to load status of FFmpeg VULKAN Graphic Cards")
            self.logger.error(traceback.format_exc())

        self.ffmpeg_gpu_infos['card_cnt'] = card_cnt
        self.ffmpeg_gpu_infos['card_infos'] = card_infos

        self._ffmpeg_card_cnt = card_cnt
        self._ffmpeg_card_infos = card_infos

    def _get_vulkan_gpus(self):
        card_infos = OrderedDict()
        card_cnt = 0
        if IS_DARWIN:
            return 0, {"0": "Apple"}
        try:
            p = Tools.popen("NcnnInfoModule", is_stderr=True)
            r = p.stderr.read()
            p.stderr.close()
            card_infos_list = list(set(re.findall("\[(\d+) (.*?)\]", r)))
            card_cnt = len(card_infos_list)
            for i in range(card_cnt):
                card_infos[card_infos_list[i][0]] = card_infos_list[i][1]
            self.logger.debug(f"VULKAN: {card_infos}")
        except:
            self.logger.error("Failed to load status of VULKAN Graphic Cards")
            self.logger.error(traceback.format_exc())

        self.vulkan_gpu_infos['card_cnt'] = card_cnt
        self.vulkan_gpu_infos['card_infos'] = card_infos

        self._vulkan_card_cnt = card_cnt
        self._vulkan_card_infos = card_infos

        for k, card in card_infos.items():
            if 'intel' in card.lower():
                self._intel_card_cnt += 1
                self._intel_card_infos[k] = card
            if 'amd' in card.lower():
                self._amd_card_cnt += 1
                self._amd_card_infos[k] = card


if __name__ == '__main__':
    _test = GraphicCardsSelector()
    print(_test.get_vulkan_gpus())
