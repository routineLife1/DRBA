# encoding=utf-8
import datetime
import json
import logging
import os

import numpy as np
import requests

from IOModule.video import VideoInfoProcessor
from Utils.ArgumentsModule import TaskArgumentManager, ArgumentManager
from Utils.StaticParameters import RT_RATIO, SR_TILESIZE_STATE, LUTS_TYPE, HDR_STATE, RESIZE_INDEX
from Utils.utils import Tools


class SvfiJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in [RT_RATIO, SR_TILESIZE_STATE, LUTS_TYPE, HDR_STATE, RESIZE_INDEX, logging.Logger]:
            return obj.name
        if type(obj) in [VideoInfoProcessor]:
            return ""
        if type(obj) in [np.ndarray]:
            return ""
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class SvfiFeedBackManager:
    def __init__(self, args: TaskArgumentManager):
        self.host = ""
        self.port = 0
        self.args = args
        self.utils = Tools()
        self.response = ""
        self.json_name = ""
        self.json_content = ""
        self.ftp_config_root = ""
        self.logger = self.args.logger
        self.validate = True
        self._initiate()

    def update_infos(self, args_dict: dict):
        sensitive_keys_list = ["app_dir", "config", "input", "output_dir", "dump_dir",
                               "gui_inputs", "rife_model_dir", "project_dir"]
        for k in sensitive_keys_list:
            if k in args_dict:
                args_dict.pop(k)
        args_dict.update({"is_steam": f"{ArgumentManager.is_steam}", "is_alpha": f"{ArgumentManager.is_alpha}",
                          "is_free": f"{ArgumentManager.is_free}", "is_demo": f"{ArgumentManager.is_demo}",
                          "is_sae": f"{ArgumentManager.is_sae}",
                          "gui_version": f"{ArgumentManager.gui_version}"})
        return args_dict

    def prepare_json(self):
        self.json_name = f"{self.utils.md5(self.args.project_dir)[:10]}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.json_content = json.dumps(self.update_infos(self.args.__dict__.copy()),
                                       cls=SvfiJsonEncoder, indent=4)

    def _initiate(self):
        self.prepare_json()

    def _clear(self):
        if os.path.exists(self.json_name):
            os.remove(self.json_name)

    def send(self):
        if not self.validate:
            self.logger.debug("Abort Sending Feedback")
            return
        data = {"log_json_name": self.json_name, "log_json": self.json_content}
        try:
            r = requests.post(f"http://{self.host}:{self.port}/SVFILogJson", data=data, timeout=10)  # 5s as timeout
            self.logger.debug(f"Send Feedback Log Success, Response: \n{r.text}")
        except:
            self.logger.debug(f"Send Feedback Log Failed")
        self._clear()


if __name__ == "__main__":
    _logger = Tools().get_logger("log_json_test", "", True)
    svfi_log_json = SvfiFeedBackManager(
        TaskArgumentManager({'input': r"D:\60-fps-Project\input or ref\Test\[2]ASCII-TEST.mp4",
                             "debug": True}))
    svfi_log_json.send()
