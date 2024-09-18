import argparse
import datetime
import functools
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from queue import Queue

import numpy as np
import psutil

from Utils.StaticParameters import appDir, RT_RATIO, SR_TILESIZE_STATE, LUTS_TYPE, HDR_STATE, RGB_TYPE, SupportFormat, \
    VFI_ANYTIME, VFI_TYPE, EncodePresetAssemply, SR_TYPE, SR_NCNN, VFI_NCNN, MAX_FRAME_CNT, TB_LIMIT, \
    RESIZE_INDEX, IS_DARWIN, DEDUP_MODE, RELEASE_TYPE, VFI_MULTI_INPUT
from Utils.utils import Tools, is_cuda_ext_ok, DefaultConfigParser
from IOModule.video import VideoInfoProcessor


class ArgumentManager:
    """Parameters from config.ini, parsed CLI, local env and compilation status

    For OLS's arguments input management. These arguments are configured according to the current running device and
    parameters from config.ini, which is not static and ought to be true variables.

    Validation of these arguments is performed at the initiation of TaskArgumentManager
    """

    # Release Version Control
    # 发布前改动以下参数即可
    is_steam = True  # True for Steam Version, False for Retail, overriden by is_sae
    release_type: RELEASE_TYPE = RELEASE_TYPE.get_release_type()
    is_alpha = False  # OBSOLETE
    is_beta = False  # OBSOLETE
    is_demo = False  # OBSOLETE, SVFI Steam DEMO, contradict to is_sae
    is_sae = False  # OBSOLETE, SAE, contradict to is_steam
    is_free = False if release_type not in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE] else True  # True for Community Version, False for Professional Version
    # PRIORITY: is_alpha = is_beta > is_sae > is_steam = is_demo > is_free
    #           Note that there are only 4 public and 1 private versions of SVFI in 2 branches
    #           Branches: Alpha, Default
    #           Public:   SVFI Steam Community, Pro, Demo; SAE
    #           Private:  SVFI Retail(Pro)

    gui_version = "6.0.10"
    ols_version = "10.0.6"
    # Title of SAE is configured in RIFE_GUI_Start.py
    version_tag = f"{gui_version}{'-alpha ' if release_type == RELEASE_TYPE.ALPHA else ''}" \
                  f"{'-beta ' if release_type == RELEASE_TYPE.BETA else ' '}" \
                  f"{'Demo ' if release_type == RELEASE_TYPE.DEMO else ''}" \
                  f"{'Professional ' if not is_free and release_type not in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE] else ''}" \
                  f"{'Community ' if is_free and release_type not in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE] else ''}" \
                  f"{'- Steam' if is_steam and release_type != RELEASE_TYPE.SAE else ''}" \
                  f"{'- Retail' if not is_steam and release_type != RELEASE_TYPE.SAE else ''}"

    update_log = f"""
    {version_tag}
    Update Log
    - fix Basic VSR++ model crashed with fp16, cpu cache mode
    - add warning for encc based encoder when output resolution can not be divided by 8
    """

    # Steam Stuff
    app_id = 1920260 if release_type == RELEASE_TYPE.DEMO else 1692080
    dlc_id = [1718750, 1991700]
    is_allow_multi_task = False  # Community Version Only, Similar should be default False

    community_qq = 264023742
    professional_qq = 1054016374

    # Function TLE Reminder
    overtime_reminder_queue = Queue()
    overtime_reminder_ids = dict()

    screen_w = 1920
    screen_h = 1080

    # poor global vars below
    # p5x_state = P5X_STATE.INACTIVE

    # demo limit
    demo_limit_frame = 1000
    preview_limit_time = 2  # seconds

    def __init__(self, args: dict):
        self.config_path = args.get("config")  # necessary

        # Basic IO info
        self.input = args.get("input", "")
        self.dump_dir = args.get("dump_dir", "")
        self.output_dir = args.get("output_dir", "")
        self.task_id = args.get("task_id", "")
        self.use_global_settings = args.get("use_global_settings", False)

        # Set task fps
        self.input_fps = args.get("input_fps", 0)
        self.target_fps = args.get("target_fps", 0)
        self.is_safe_fps = args.get("is_safe_fps", True)

        # Set io info
        self.input_ext = ".mp4"
        self.output_ext = args.get("output_ext", ".mp4")
        self.is_img_input = False
        self.is_img_output = args.get("is_img_output", False)
        self.is_keep_cache = args.get("is_keep_cache", False)
        self.is_save_audio = args.get("is_save_audio", True)
        self.is_save_subs = args.get("is_save_subs", True)
        self.input_start_point = args.get("input_start_point", None)
        self.input_end_point = args.get("input_end_point", None)
        if self.input_start_point == "00:00:00":
            self.input_start_point = None
        if self.input_end_point == "00:00:00":
            self.input_end_point = None
        self.output_chunk_cnt = args.get("output_chunk_cnt", 0)
        self.interp_start = args.get("interp_start", 0)
        self.risk_resume_mode = args.get("risk_resume_mode", False)

        # Set scene detection
        self.is_no_scdet = args.get("is_no_scdet", False)
        self.is_scdet_mix = args.get("is_scdet_mix", False)
        self.use_scdet_fixed = args.get("use_scdet_fixed", False)
        self.is_scdet_output = args.get("is_scdet_output", False)
        self.scdet_threshold = args.get("scdet_threshold", 12) + 2
        self.pure_scene_threshold = args.get("pure_scene_threshold", 10)
        self.scdet_max_threshold = args.get("scdet_max_threshold", 80)
        self.scdet_flow_cnt = args.get("scdet_flow_cnt", 4)
        self.scdet_mode = args.get("scdet_mode", 0)
        self.scene_list_path = args.get("scene_list_path", r"")
        if self.scene_list_path is not None and not len(self.scene_list_path):
            self.scene_list_path = None
        self.scene_list_offset = args.get("scene_list_offset", 0)
        self.scene_list_ratio = args.get("scene_list_ratio", 2.0)  # output_fps in TC / original_fps
        # so that scene_list is imported at original video fps: scene fps = original fps

        # Set Dedup
        self.remove_dup_mode = DEDUP_MODE(args.get("remove_dup_mode", 1))  # default: TruMotion
        self.remove_dup_threshold = args.get("remove_dup_threshold", 0.65)
        self.use_dedup_sobel = args.get("use_dedup_sobel", False)
        self.use_dedup_flow = args.get("use_dedup_flow", False)

        # Set Manual Memory Buffer
        self.use_manual_buffer = args.get("use_manual_buffer", False)
        self.manual_buffer_size = args.get("manual_buffer_size", 1)

        # Set Resize
        self.resize_settings_index = RESIZE_INDEX(args.get("resize_settings_index", 0))  # Default as Custom
        resize_width = Tools.get_plural(args.get("resize_width", 0))
        resize_height = Tools.get_plural(args.get("resize_height", 0))
        self.resize_param = [resize_width, resize_height]  # resize parameter, 输出分辨率参数,
        # this param will always be updated in _update_frame_size
        self.resize_exp = args.get("resize_exp", 0)  # actually it's resize_settings_index
        self.transfer_ratio = RT_RATIO(args.get("transfer_ratio_index", 0))

        # Set Cropper
        self.is_auto_crop = args.get("is_auto_crop", False)
        crop_width = args.get("crop_width", 0)
        crop_height = args.get("crop_height", 0)
        self.crop_param = [crop_width, crop_height]  # crop parameter, relevant to output resolution only.
        self.is_pad_crop = args.get("is_pad_crop", False)

        # Set Super Resolution
        self.use_sr = args.get("use_sr", False)
        self.use_sr_algo = args.get("use_sr_algo", "")
        self.use_sr_model = args.get("use_sr_model", "")
        self.use_sr_mode = args.get("use_sr_mode", "")
        self.use_sr_gpu = max(0, args.get("use_sr_gpu", 0))  # prevent -1
        self.sr_thread_cnt = args.get("sr_thread_cnt", 1)

        self.sr_tilesize_mode = SR_TILESIZE_STATE(args.get("sr_tilesize_mode", 0))
        self.sr_tilesize = args.get("sr_tilesize", 200)
        if self.sr_tilesize_mode != SR_TILESIZE_STATE.CUSTOM:
            self.sr_tilesize = SR_TILESIZE_STATE.get_tilesize(self.sr_tilesize_mode)
        self.sr_realCUGAN_tilemode = args.get("sr_realcugan_tilemode", 2)  # default: h, w both /2
        self.sr_realCUGAN_low_vram_mode = args.get("sr_realcugan_low_vram_mode", 0)  # default: None
        self.sr_realCUGAN_alpha = args.get("sr_realcugan_alpha", 1)  # default: None
        self.sr_ncnn_denoise = args.get("sr_ncnn_denoise", 0)  # default: None
        self.sr_ncnn_tta = args.get("sr_ncnn_tta", False)
        self.sr_module_exp = args.get("sr_module_exp", 0)
        self.sr_seq_len = int(args.get("sr_seq_len", 5))  # for RealBasicVSR, BasicVSR++
        self.use_realesr_fp16 = args.get("use_realesr_fp16", False)
        self.is_old_basicvsrpp = args.get("is_old_basicvsrpp", True)
        self.is_restore_first = args.get("is_restore_first", False)
        self.is_sr_after_vfi = args.get("is_sr_after_vfi", False) and args.get("use_sr", False)

        # Set AI Restore
        self.use_fmnet_hdr = args.get("use_fmnet_hdr", False)  # modify model in advanced settings
        self.use_deep_deband = args.get("use_deep_deband", False)  # modify model in advanced settings

        self.use_restore = args.get("use_restore", False)  # TODO: Still under prototype
        self.use_restore_algo = args.get("use_restore_algo", "")
        self.use_restore_model = args.get("use_restore_model", "")

        # Set Render Settings
        self.render_gap = args.get("render_gap", 1000)
        self.use_crf = args.get("use_crf", True)
        self.use_bitrate = args.get("use_bitrate", False)
        self.render_crf = args.get("render_crf", 12)
        self.render_bitrate = args.get("render_bitrate", 90)
        self.render_encoder = args.get("render_hwaccel_mode", "AUTO")
        self.render_encode_format = args.get("render_encoder", "AUTO")
        self.render_encoder_preset = args.get("render_encoder_preset", "AUTO")
        self.use_render_avx512 = args.get("use_render_avx512", False)
        self.use_render_zld = args.get("use_render_zld", False)  # enables zero latency decode
        self.render_nvenc_preset = args.get("render_hwaccel_preset", "")
        self.use_hwaccel_decode = args.get("use_hwaccel_decode", True)
        self.use_manual_encode_thread = args.get("use_manual_encode_thread", False)
        self.render_encode_thread = args.get("render_encode_thread", 16)
        self.use_render_encoder_default_preset = args.get("use_render_encoder_default_preset", False)
        self.is_encode_audio = args.get("is_encode_audio", False)
        self.is_quick_extract = args.get("is_quick_extract", True)
        self.hdr_cube_mode = LUTS_TYPE(args.get("hdr_cube_index", 0))
        self.use_vspipe_decode = args.get("use_vspipe_decode", False)
        self.is_16bit_workflow = args.get("is_16bit_workflow", False)
        self.hdr_mode = args.get("hdr_mode", 0)  # dilapidated
        # if self.hdr_mode == 0:  # AUTO
        #     self.hdr_mode = HDR_STATE.AUTO
        # else:
        #     self.hdr_mode = HDR_STATE(self.hdr_mode)
        self.is_hdr_strict = args.get("is_hdr_strict", True)
        self.is_dv_with_hdr10 = args.get("is_dv_with_hdr10", True)
        if self.hdr_cube_mode != LUTS_TYPE.NONE or self.use_fmnet_hdr:
            self.is_hdr_strict = False

        self.render_customized = args.get("render_customized", "").strip('"').strip("'")
        self.decode_customized = args.get("decode_customized", "").strip('"').strip("'")
        self.is_no_concat = args.get("is_no_concat", False)

        # Set Auxiliary Render Settings
        self.use_fast_denoise = args.get("use_fast_denoise", False)
        self.use_fast_grain = args.get("use_fast_grain", False)  # only available with VSPipe
        self.is_loop = args.get("is_loop", False)
        self.is_render_slow_motion = args.get("is_render_slow_motion", False)
        self.render_slow_motion_fps = args.get("render_slow_motion_fps", 0)
        self.use_deinterlace = args.get("use_deinterlace", False)
        self.use_depan = args.get("use_depan", False)
        # self.is_keep_head = args.get("is_keep_head", False)

        # Set Hardware Status
        self.cuda_card_cnt = args.get("cuda_card_cnt", 0)
        self.intel_card_cnt = args.get("intel_card_cnt", 0)
        self.amd_card_cnt = args.get("amd_card_cnt", 0)

        # Set VFI
        self.use_ncnn = args.get("use_ncnn", False)
        self.ncnn_thread = args.get("ncnn_thread", 4)  # dilapidated
        self.vfi_thread_cnt = args.get("vfi_thread_cnt", 1)
        self.ncnn_gpu = args.get("ncnn_gpu", 0)  # dilapidated
        self.rife_tta_mode = args.get("rife_tta_mode", 0)
        self.rife_tta_iter = args.get("rife_tta_iter", 1)
        self.use_evict_flicker = args.get("use_evict_flicker", False)
        self.use_vfi_fp16 = args.get("use_vfi_fp16", False)
        self.rife_scale = args.get("rife_scale", 1.0)
        self.vfi_algo = args.get("vfi_algo", "ncnn_rife")
        self.vfi_model = args.get("vfi_model", "rife-v4")
        self.rife_exp = args.get("rife_exp", 2.0)  # 补帧倍率
        self.is_rife_reverse = args.get("is_rife_reverse", False)
        self.use_vfi_gpu = max(0, args.get("use_vfi_gpu", 0))  # !
        VFI_TYPE.update_current_gpu_id(self.use_vfi_gpu)
        self.use_rife_auto_scale = args.get("use_rife_auto_scale", False)
        self.rife_interp_before_resize = args.get("rife_interp_before_resize", 0)
        self.use_rife_forward_ensemble = args.get("use_rife_forward_ensemble", False)
        self.rife_interlace_inference = args.get("rife_interlace_inference", 0)
        self.vfi_mask = os.path.join(appDir, "log", f"mask_{self.task_id}.png")
        if not os.path.exists(self.vfi_mask):
            self.vfi_mask = ""
        self.vfi_recon_iter = args.get("vfi_recon_iter", 2)

        # For RIFE-Plus Model Only, it's actually switch of fastmode
        self.rife_layer_connect_mode = args.get("rife_layer_connect_mode", 0)

        self.use_true_motion = args.get("use_true_motion", True)  # OBSOLETE VARIABLE SINCE SVFI 3.31.0-alpha

        # GPU Usages
        self.use_multi_gpus = args.get("use_multi_gpus", False)
        self.use_trt_int8 = args.get("use_trt_int8", False)

        # SVFI Preset
        self.svfi_preset = args.get("svfi_preset", 2)  # default medium

        self.debug = args.get("debug", False)
        self.is_test_mode = args.get("is_test_mode", False)
        self.multi_task_rest = args.get("multi_task_rest", False)  # obsolete
        self.multi_task_rest_interval = args.get("multi_task_rest_interval", 1)
        self.after_mission = args.get("after_mission", 0)  # obsolete, gui only
        self.force_cpu = args.get("force_cpu", False)
        self.dumb_mode = args.get("dumb_mode", True)
        self.is_no_dedup_render = args.get("is_no_dedup_render", True)

        # OLS Mission Type
        self.concat_only = args.get("concat_only", False)
        self.extract_only = args.get("extract_only", False)
        self.render_only = args.get("render_only", False)
        self.is_preview = args.get("is_preview", False)
        self.version = args.get("version", "0.0.0 beta")

        # Set Pipe Settings
        self.is_pipe_in = args.get("is_pipe_in", False)  # As middle piper if True
        self.is_pipe_out = args.get("is_pipe_out", False)  # As middle piper if True
        self.is_pipe_rgb = args.get("is_pipe_rgb", False)
        self.pipe_in_fps = args.get("pipe_in_fps", 24)
        self.pipe_in_width = args.get("pipe_in_width", 0)
        self.pipe_in_height = args.get("pipe_in_height", 0)
        self.pipe_pix_fmt = args.get("pipe_pix_fmt", "rgb24")  # channel = 3
        self.pipe_colormatrix = args.get("pipe_colormatrix", "709")  # no use

        # Preview Imgs
        self.is_preview_imgs = args.get("is_preview_imgs", True)
        self.is_preview_pipe = args.get("is_preview_pipe", False)

        # Update Dilapidated Keys from Presets
        self.update_dilapidated_keys(args)
        self.update_platform_rules()

    def update_dilapidated_keys(self, args: dict):
        # Update VFI Parameters
        # if args.get("rife_model", "") != "":  # 2022.06.28
        #     _old_path = args["rife_model"]
        #     self.vfi_algo = os.path.basename(os.path.dirname(_old_path))
        #     self.vfi_model = os.path.basename(_old_path)
        pass
        # Only available at UI layer

    def update_platform_rules(self):
        """
        Disable related keys due to platform support
        :return:
        """
        if IS_DARWIN:
            self.is_preview_imgs = False

    @staticmethod
    def is_empty_overtime_task_queue():
        return ArgumentManager.overtime_reminder_queue.empty()

    @staticmethod
    def put_overtime_task(_over_time_reminder_task):
        ArgumentManager.overtime_reminder_queue.put(_over_time_reminder_task)

    @staticmethod
    def get_overtime_task():
        return ArgumentManager.overtime_reminder_queue.get()

    @staticmethod
    def update_screen_size(w: int, h: int):
        ArgumentManager.screen_h = h
        ArgumentManager.screen_w = w

    @staticmethod
    def get_screen_size():
        """

        :return: h, w
        """
        return ArgumentManager.screen_h, ArgumentManager.screen_w

    # Below is for Steam DLC Content
    @staticmethod
    def allow_multi_task():
        ArgumentManager.is_allow_multi_task = True


class TaskArgumentManager(ArgumentManager):
    """
        For OLS's current input's arguments validation
    """

    def __init__(self, _args: dict):
        super().__init__(_args)
        # initiate logger
        self.logger = Tools.get_logger("CLI", appDir, debug=self.debug)

        #
        # Declare Task Arguments default values (not in ArgumentsManager)
        #
        self.project_name = f"{Tools.get_filename(self.input)}_{self.task_id}"
        self.project_dir = self.output_dir

        self.video_hdr_mode = HDR_STATE.NOT_CHECKED
        self.is_video_dv_hdr10 = False

        self.interp_times = 0  # self.proc_fps / self.input_fps, rounded
        self.all_frames_cnt = 0
        self.proc_fps = 0  # SVFI内部处理的帧率，会在render时降采样；仅限ffmpeg系时使用
        self.frames_queue_len = 0
        self.dup_skip_limit = 0
        self.target_fps_frac = (0, 1)
        self.is_p5x_mode = False  # OBSOLETE VARIABLE SINCE SVFI 3.31.0-alpha
        self.p5x_mode_cword = 2
        self.is_vfi_gen_ts_frames = False  # is vfi generating assigned timesteps
        self.is_reader_use_input_fps = False
        self.task_info = {"chunk_cnt": 0, "render": 0,
                          "read_now_frame": 0,
                          "rife_now_frame": 0,
                          "recent_scene": 0, "scene_cnt": 0,
                          "decode_process_time": 0,
                          "render_process_time": 0,
                          "rife_task_acquire_time": 0,
                          "rife_process_time": 0,
                          "rife_queue_len": 0,
                          "sr_now_frame": 0,
                          "sr_task_acquire_time": 0,
                          "sr_process_time": 0,
                          "sr_queue_len": 0, }  # 有关任务的实时信息

        self.concat_final_ext = '.mp4'  # used when assigned to output gif or webp or so
        self.concat_final_filename = ""
        self.is_final_output_ext_changed = False

        # Workflow passed flags
        self.is_vfi = not (self.render_only or self.extract_only or self.concat_only)
        self.is_vfi_flow_passed = self.is_vfi and \
                                  (VFI_TYPE.get_model_version(self.vfi_model) != VFI_TYPE.TensorRT)  # vpy rife46
        self.is_sr_flow_passed = False  # updated in SR workflow _load_sr_module
        self.is_restore_flow_passed = False  # updated in SR workflow _load_rs_module
        self.is_read_flow_apply_lut = self.hdr_cube_mode != LUTS_TYPE.NONE and \
                                      self.render_encoder not in EncodePresetAssemply.encc_encoders

        # Preview
        self.preview_imgs = list()

        self.ffmpeg = Tools.fillQuotation(os.path.join(appDir, "ffmpeg"))
        self.mkvmerge = Tools.fillQuotation(os.path.join(appDir, "mkvmerge"))

        self.main_error = list()

        #
        # Update Input Output Arguments
        #
        self._update_project_io_args()
        self.video_info_instance = VideoInfoProcessor(input_file=self.input, logger=self.logger,
                                                      project_dir=self.project_dir,
                                                      hdr_cube_mode=self.hdr_cube_mode,
                                                      use_ai_pq_hdr=self.use_fmnet_hdr,

                                                      find_crop=self.is_auto_crop,
                                                      crop_param=self.crop_param,

                                                      is_pipe_in=self.is_pipe_in,
                                                      pipe_in_height=self.pipe_in_height,
                                                      pipe_in_width=self.pipe_in_width,
                                                      pipe_in_fps=self.pipe_in_fps)
        self._update_hdr_mode()
        self._re_update_input_args()
        self._update_io_ext()
        self._update_frame_size()
        self._update_crop_param()

        #
        # Settings below is based on input's information
        # many arguments are modified accordingly
        #
        self._update_auto_presets()

        self._update_fps_args()
        self._update_input_section()
        self._update_audio_mux_args()
        self._update_frames_cnt()
        self._update_render_settings()
        self._update_dedup_settings()
        self._update_workflow_precision()
        self._update_task_queue_size_by_memory()
        self._update_task_thread_num()

        # Check Initiation Info
        if self.release_type == RELEASE_TYPE.DEMO:
            self.logger.warning(f"You are running the demo version of SVFI, "
                                f"the length of output will be limited within {self.demo_limit_frame} frames.")

        self.logger.info(
            f"Project Info Summary: "
            f"FPS: {self.input_fps:.3f} -> {self.proc_fps:.3f} -> {self.target_fps:.3f}, FRAMES_CNT: {self.all_frames_cnt}, "
            f"INTERP_TIMES: {self.interp_times}, "
            f"HDR: {self.video_hdr_mode.name}, FRAME_SIZE: {self.frame_size}, QUEUE_LEN: {self.frames_queue_len}, "
            f"INPUT_EXT: {self.input_ext}, OUTPUT_EXT: {self.output_ext}")

    def _update_project_io_args(self):
        """
        Check and update:
            - Input path
            - Output Directory
            - Final Output Path
            - Project Folder
            - is_image_output
            - is_keep_cache
        According to:
            - Input File Type
            - Output Dir
        :return:
        """
        # Check Input
        if not len(self.input):
            raise OSError("Input Path is empty, Program will not proceed")

        # Check and update Output Dir
        if not len(self.output_dir):
            # 未填写输出文件夹
            self.output_dir = os.path.dirname(self.input)
        else:
            """
            case 0 (normal): args.output_dir is output_dir
            case 1: args.output_dir is output_dir but not exists
            case 2: args.output_dir is output_path
            """
            output_dir_dir = os.path.dirname(self.output_dir)
            if not os.path.exists(self.output_dir):  # e.g. output/output.mkv
                output_name, output_ext = os.path.splitext(os.path.basename(self.output_dir))  # e.g. output.mkv
                if not len(output_ext):
                    # case 1
                    if not os.path.exists(output_dir_dir):
                        raise OSError("Cannot determine output path, please check the existence of output directory")
                else:
                    # case 2
                    self.concat_final_filename = output_name  # output dir is actually a filename
                    self.output_dir = os.path.dirname(self.output_dir)  # update output dir
                    self.logger.info(f"Output filename has been changed to {self.concat_final_filename}")
            elif os.path.isfile(self.output_dir):
                self.output_dir = os.path.dirname(self.output_dir)
                self.logger.info(f"Output folder has been changed to {self.output_dir}")

        # Extract-Only Mode
        if self.extract_only and self.output_ext not in SupportFormat.img_outputs:
            self.is_img_output = True
            self.output_ext = ".png"
            self.logger.warning("Output extension is changed to .png")

        # Update Project Folder
        if not len(self.dump_dir) or self.is_img_output:
            self.project_dir = os.path.join(self.output_dir, self.project_name)
        else:
            self.project_dir = os.path.join(self.dump_dir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        sys.path.append(self.project_dir)

        # Update Image output folder
        # Image output is directly written to output_dir, with no respect to project_dir
        if self.is_img_output:
            self.output_dir = os.path.join(self.output_dir, self.project_name)

        # assure existence of output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Update whether to keep project dir
        if self.is_scdet_output and self.is_vfi:
            self.is_keep_cache = True

        # Write config path to project dir
        config_name = os.path.split(self.config_path)[-1]
        config_path = os.path.join(self.project_dir, config_name)
        shutil.copy(self.config_path, config_path)
        with open(os.path.join(self.project_dir, "task_restore.json"), "w", encoding='utf-8') as f:
            restore_info = {"task_id": self.task_id, "config_path": config_path, "input_path": self.input}
            json.dump(restore_info, f, indent=4)

        self.logger.info(f"Init New Project: {self.input} -> {self.project_dir} -> {self.output_dir}")

    def _update_hdr_mode(self):
        """

        hdr_mode in ArgumentManager is dilapidated
        all conditions are modified according to hdr_mode of video_info_instance, using video_hdr_mode

        :return:
        """
        # if self.hdr_mode == HDR_STATE.AUTO:  # Auto
        #     self.hdr_mode = self.video_info_instance.hdr_mode
        # no hdr at -1, 0 checked and None, 1 hdr, 2 hdr10, 3 DV, 4 HLG
        # hdr_check_status indicates the final process mode for (hdr) input
        self.video_hdr_mode = self.video_info_instance.hdr_mode
        self.logger.info(f"Auto Sets HDR mode to {self.video_hdr_mode.name}")
        self.is_video_dv_hdr10 = self.video_info_instance.is_video_dv_hdr10
        # authentic hdr mode of input video, not affected by settings

    def _re_update_input_args(self):
        """
        Update Input arguments based on self.video_info_instance
        :return:
        """
        self.is_img_input = self.video_info_instance.is_img_input  # update input file type
        self.input = self.video_info_instance.input_file  # update true input file, used when input is single image

    def _update_io_ext(self):
        # update extension
        self.input_ext = os.path.splitext(self.input)[1] if os.path.isfile(self.input) else ""
        self.input_ext = self.input_ext.lower()
        if not self.output_ext.startswith('.'):
            self.output_ext = "." + self.output_ext
        if "ProRes" in self.render_encode_format and not self.is_img_output:
            self.output_ext = ".mov"
        if self.is_img_output and self.output_ext not in SupportFormat.img_outputs:
            self.output_ext = ".png"
        self.concat_final_ext = self.output_ext
        if self.output_ext in ['.gif', '.webp']:
            self.output_ext = '.mp4'
            self.is_final_output_ext_changed = True
            self.is_save_audio = False
            self.is_save_subs = False

    def _update_auto_presets(self):
        auto_argument_manager = AutoArgumentManager(self)
        auto_argument_manager.modify()

    def _update_fps_args(self):
        """
        Update input, process, and output fps
        :return:
        """
        # set target fps by user settings
        if not self.is_img_input:  # 输入不是文件夹，使用检测到的帧率
            is_manual_input_fps = abs(self.video_info_instance.fps - self.input_fps) > 0.5
            if not self.is_safe_fps and not self.use_global_settings and is_manual_input_fps and self.input_fps > 0:
                self.input_fps = self.input_fps
                self.logger.warning(f"Input FPS has been changed to {self.input_fps}, which is assigned manually. "
                                    f"The detected input fps is {self.video_info_instance.fps}. "
                                    f"Please note that this may result in VA out of sync. "
                                    f"If this is the case, leave input fps to 0")
            else:
                self.input_fps = self.video_info_instance.fps
        elif not self.input_fps:  # 输入是文件夹，使用用户的输入帧率; 用户有毒，未发现有效的输入帧率
            if self.is_img_input and self.is_img_output:
                self.logger.info("Input and output are both images, input fps is set to 1")
                self.input_fps = 1
            else:
                raise OSError("Not Find Input FPS, Input File is not valid")
        # set target fps by user settings
        if self.render_only:
            self.target_fps = self.input_fps
            self.logger.info(f"Render only, target fps is changed to input fps: {self.target_fps}")
        else:
            if not self.target_fps:  # 未找到用户的输出帧率
                self.target_fps = self.rife_exp * self.input_fps  # default
            if self.target_fps + 0.1 < self.input_fps:
                raise OSError(f"Output FPS {self.target_fps} should be higher than Input FPS {self.input_fps}")
        # update target fps and proc fps by TruMotion Prerequisites Check
        # proc fps should be within (target_fps, input_fps)
        ratio = self.target_fps / self.input_fps  # e.g. 2.5
        self.interp_times = round(ratio + 1e-4)
        self.proc_fps = self.target_fps

        # update target fps by HDR Check
        is_fps_complied_to_hdr = self.video_hdr_mode in [HDR_STATE.DOLBY_VISION, HDR_STATE.HDR10_PLUS]
        if is_fps_complied_to_hdr:
            self.proc_fps = self.interp_times * self.input_fps
            self.target_fps = self.proc_fps
            self.logger.info(f"Complying with HDR10+ / DV, target fps is changed to {self.target_fps}")
        if not self.is_vfi_flow_passed:
            # VSPipe handles fps or keep the same
            self.input_fps = self.input_fps * self.interp_times  # rife 4.6
            self.proc_fps = self.input_fps
            self.target_fps = self.proc_fps  # TODO: TruMotion in vapoursynth

        if self.remove_dup_mode in [DEDUP_MODE.RESAMPLE, DEDUP_MODE.TRUMOTION] and self.is_vfi_flow_passed:
            is_compatible = True
            reason = ""
            ratio_d = ratio * 2
            if self.remove_dup_mode == DEDUP_MODE.TRUMOTION and abs(abs(round(ratio, 2) - round(ratio)) - 0.5) >= 1e-1:
                is_compatible = False
                reason = "ratio = output/input_fps does not satisfy: abs(abs(round(ratio, 2) - round(ratio)) - 0.5) < 1e-1"
            elif self.remove_dup_mode == DEDUP_MODE.RESAMPLE and abs(round(ratio_d, 2) - round(ratio_d)) >= 1e-1:
                is_compatible = False
                reason = "ratio = output/input_fps * 2 does not satisfy: abs(round(ratio_d, 2) - round(ratio_d)) < 1e-1"
            elif VFI_TYPE.get_model_version(self.vfi_model) not in VFI_ANYTIME:
                is_compatible = False
                reason = "incompatible with current VFI model, switch to another"
            elif self.is_scdet_mix:
                is_compatible = False
                reason = "incompatible with Scene Mixing"
            elif is_fps_complied_to_hdr:
                is_compatible = False
                reason = "incompatible with Input HDR format"

            if not is_compatible:
                self.logger.warning(f"Unable to use dedup: {reason}")
                self.remove_dup_mode = DEDUP_MODE.NONE
            else:
                # meet p5x mode requirement
                self.proc_fps = self.input_fps  # e.g. 24fps
                self.p5x_mode_cword = int(ratio)  # TRUMOTION, e.g. 2

        # TruMotion or FirstOrderDedup
        self.is_vfi_gen_ts_frames = self.remove_dup_mode in [DEDUP_MODE.DIFF, DEDUP_MODE.TRUMOTION]
        self.is_reader_use_input_fps = self.remove_dup_mode in [DEDUP_MODE.RESAMPLE, DEDUP_MODE.TRUMOTION]

        # update target fps frac
        self.target_fps_frac = self.video_info_instance.fps_frac

        # WARNING: only support 1.001 or 1 as denomitor for fps (since SVFI 3.14.8)
        def check_float(fps):
            return abs(fps - round(fps)) > 1e-4

        if check_float(self.input_fps):  # input fps 23.976
            is_ntsc_fps = False
            if not check_float(self.target_fps):  # output fps 60, final output: 59.94
                if self.is_safe_fps:
                    self.target_fps_frac = (round(self.target_fps) * 1000, 1001)
                    self.target_fps /= 1.001
                    is_ntsc_fps = True
                else:  # output fps 60, final output: 60
                    self.target_fps_frac = (round(self.target_fps), 1)
                if not self.is_reader_use_input_fps:
                    self.proc_fps = self.target_fps  # update proc fps to avoid 23.976 -> 60 -> 59.94
            else:  # output fps 59.94, final output: 59.94
                if self.is_safe_fps:
                    self.target_fps_frac = (round(self.target_fps * 1.001) * 1000, 1001)
                    is_ntsc_fps = True
                else:
                    self.target_fps_frac = (round(self.target_fps * 1000), 1000)
                    self.logger.info(
                        f"Unsafe fps settings found, input={self.input_fps:.3f}, output={self.target_fps:.3f}")
            if is_ntsc_fps:
                self.logger.info(f"Modify output fps to {self.target_fps:.2f} as NTSC standard")
        else:  # input fps 24
            if not check_float(self.target_fps):  # output fps 60, final output: 60
                self.target_fps_frac = (round(self.target_fps), 1)
            else:  # output fps 59.94, final output: 60
                if self.is_reader_use_input_fps:  # input 25, output 62.5
                    self.target_fps_frac = (round(self.target_fps * 1000), 1000)
                else:
                    self.logger.warning("It's impossible to output Non-integer FPS with input FPS in integer: "
                                        f"{round(self.input_fps)} -x-> {self.target_fps:.2f}, "
                                        f"auto change output fps to {round(self.target_fps * 1.001)}")
                    self.target_fps = round(self.target_fps * 1.001)  # input 62.5, final output 63
                    self.target_fps_frac = (round(self.target_fps), 1)

    def _update_input_section(self):
        # update demo limit
        if self.release_type == RELEASE_TYPE.DEMO:
            self.input_start_point = "00:00:00"  # "%H:%M:%S"
            delta = datetime.timedelta(seconds=int(self.demo_limit_frame / self.target_fps))
            self.input_end_point = str(delta)  # "%H:%M:%S"
        if self.is_preview:
            delta = datetime.timedelta(seconds=self.preview_limit_time)
            self.input_end_point = str(delta)  # "%H:%M:%S"

    def _update_audio_mux_args(self):
        # update whether mux audio
        if not len(self.video_info_instance.ffmpeg_audio_info) and self.is_save_audio:
            self.logger.warning("No Audio input detected, disable audio muxing")
            self.is_save_audio = False

    def _update_frames_cnt(self):
        # update all frames cnt
        if self.is_img_input:
            self.all_frames_cnt = round(self.video_info_instance.frames_cnt * self.target_fps / self.input_fps)
        else:
            self.all_frames_cnt = abs(int(self.video_info_instance.duration * self.target_fps))
        if not self.all_frames_cnt:
            self.logger.warning("SVFI can't estimate task speed since couldn't obtain number of frames")
        if self.all_frames_cnt > MAX_FRAME_CNT:
            raise OSError(f"SVFI can't afford input exceeding {MAX_FRAME_CNT} frames")

        tmp_start_point, tmp_end_point = self.get_process_clip_points()
        tmp_start_point = tmp_start_point if tmp_start_point is not None else 0
        tmp_end_point = tmp_end_point if tmp_end_point is not None else self.video_info_instance.duration
        if tmp_end_point > 0 and tmp_end_point > tmp_start_point:
            clip_duration = tmp_end_point - tmp_start_point
            clip_fps = self.target_fps
            self.all_frames_cnt = min(self.all_frames_cnt, round(clip_duration * clip_fps))
        self.logger.info(
            f"Update Input Section: in {tmp_start_point} ({self.input_start_point}) -> "
            f"out {tmp_end_point:.2f} ({self.input_end_point}), "
            f"all_frames_cnt = {self.all_frames_cnt}, "
            f"start_frame = {self.interp_start}, start_chunk = {self.output_chunk_cnt}")

    def _update_render_settings(self):
        # If is preview, use cpu render compulsorily
        if self.is_preview:
            self.render_customized = ""
            self.render_encoder = "CPU"
            self.render_encode_format = "H264,8bit"
            self.render_encoder_preset = "fast"
            self.dumb_mode = False

        # update Render Chunk Size
        if self.render_gap > self.proc_fps:
            self.render_gap -= round(self.render_gap % round(self.proc_fps))
            self.logger.debug(f"To sync with the target fps, Update Render Gap (Chunk Size) to {self.render_gap}")

        # check if output resolution is compatible with render encoder
        if re.search("encc", self.render_encoder, re.I):
            if self.dumb_mode and not self.is_resolution_compatible_with_vs():
                self.logger.warning(f"Current render encoder is not compatible with "
                                    f"input resolution {self.resize_param}, which should be divided by 8, "
                                    "switch to CPU encoder")
                self.render_encoder = "CPU"
                self.render_encode_format = "H264,8bit"
                self.render_encoder_preset = "fast"

    def _update_dedup_settings(self):
        if self.is_pipe_in and self.remove_dup_mode in [DEDUP_MODE.D2, DEDUP_MODE.D3]:
            self.logger.warning("Pipe mode does not support 1xn dedup mode, dedup disabled")
            self.remove_dup_mode = DEDUP_MODE.NONE

        vfi_model_version = VFI_TYPE.get_model_version(self.vfi_model)
        if self.remove_dup_mode == DEDUP_MODE.RECON:
            if vfi_model_version == VFI_TYPE.GmfSs and "_real" in self.vfi_model:
                self.logger.warning("GMFSS real does not support Dual Referenced Reconstruction, dedup disabled")
                self.remove_dup_mode = DEDUP_MODE.NONE
            if vfi_model_version in VFI_MULTI_INPUT or vfi_model_version not in VFI_ANYTIME:
                self.logger.warning("Current model does not support Dual Referenced Reconstruction, dedup disabled")
                self.remove_dup_mode = DEDUP_MODE.NONE

        # Update Interp Mode Info
        if self.remove_dup_mode == DEDUP_MODE.SINGLE:  # 单一模式
            self.remove_dup_threshold = max(0.01, self.remove_dup_threshold)
        else:  # 0， 不去除重复帧
            self.remove_dup_threshold = 0.001

    def _update_workflow_precision(self):
        """Update Workflow Precision Status based on functions enabled

        :return:
        """
        # update fp16 inference strategy
        # if self.is_vfi_flow_passed and self.use_realesr_fp16:
        #     self.use_realesr_fp16 = False
        #     self.logger.warning("Automatically disable SR/Restore fp16 procedure to adapt to VFI")
        if (self.is_vfi_flow_passed and re.search("trt|onnx", self.vfi_model, re.I)) or \
            (self.use_sr and re.search("trt|onnx", self.use_sr_model, re.I)):
            if not self.use_realesr_fp16:
                self.logger.info("Automatically enable sr fp16 inference to adapt to TensorRT-based model")
            if not self.use_vfi_fp16:
                self.logger.info("Automatically enable vfi fp16 inference to adapt to TensorRT-based model")
            self.use_vfi_fp16 = True
            self.use_realesr_fp16 = True

        elif self.is_vfi_flow_passed and \
                (self.use_sr or self.use_fmnet_hdr or self.use_deep_deband) and \
                self.use_realesr_fp16:
            self.use_realesr_fp16 = False
            # self.logger.warning("Automatically enable VFI fp16 to adapt to SR fp16 procedure. "
            #                     "If output contains blurred bar using VFI, "
            #                     "disable Half-Precision mode of SR or disable VFI")
            self.logger.warning("Automatically disable SR/Restore fp16 procedure due to usage of VFI")

        # update 16bit workflow status
        is_pipe_in_8bit = any([i in self.pipe_pix_fmt for i in ['rgb24', 'gbrp']])
        is_pipe_in_16bit = not is_pipe_in_8bit
        if not self.is_16bit_workflow and \
                self.is_pipe_in and is_pipe_in_16bit:
            self.is_16bit_workflow = True
            self.logger.info("Automatically enable high precision workflow to adapt 16bit input")
        if not self.is_16bit_workflow:
            RGB_TYPE.change_8bit(True)
            return
        is_disable_16bit = False
        reason = ""
        if self.is_img_input:
            is_disable_16bit = True
            reason = "Image Sequence Input"
        if self.use_sr and SR_TYPE.get_model_version(self.use_sr_algo) in SR_NCNN:
            is_disable_16bit = True
            reason = "SR NCNN-based Model"
        if not (self.render_only or self.extract_only) and VFI_TYPE.get_model_version(self.vfi_model) in VFI_NCNN:
            is_disable_16bit = True
            reason = "VFI NCNN-based Model"
        if self.is_test_mode:
            is_disable_16bit = True
            reason = "Test Mode"
        if self.is_pipe_in and is_pipe_in_8bit:
            is_disable_16bit = True
            reason = "8bit Pipe Input"

        if is_disable_16bit:
            if self.is_pipe_in and is_pipe_in_16bit:
                # cannot satisfy low precision requirement:
                raise OSError("Current Settings do not support High Precision Workflow, "
                              "please change input pixel format to rgb24")
            self.is_16bit_workflow = False
            RGB_TYPE.change_8bit(True)
            self.logger.warning(f"High Precision Workflow disabled, Cause: {reason}")

    def _update_frame_size(self):
        # 规整化输出输入分辨率
        self.frame_size = (round(self.video_info_instance.frame_size[0]),
                           round(self.video_info_instance.frame_size[1]))
        self.first_frame_size = (round(self.video_info_instance.first_img_frame_size[0]),
                                 round(self.video_info_instance.first_img_frame_size[1]))
        if self.resize_settings_index != RESIZE_INDEX.CUSTOM:  # need update resize param
            # For Image Sequence input, resize param === 0, while first_frame_size != 0
            if self.is_pipe_in:
                self.logger.warning(f"Pipe input does not support custom resize")
            self.resize_param = RESIZE_INDEX.update_resize_params(self.resize_settings_index, self.frame_size)
            if self.video_info_instance.is_non_square_pixel:
                w, h = self.frame_size
                if h > 0:
                    far = w / h  # "frame aspect ratio"
                    w, h = self.resize_param
                    w = round(w * self.video_info_instance.display_ratio / far) // 2 * 2
                    self.logger.warning(f"DAR missed, update output resolution from {self.resize_param} to {w}x{h}")
                    self.resize_param = w, h

            self.logger.debug(f"Update Output Resolution: {self.resize_param}")

    def _update_crop_param(self):
        if self.is_auto_crop:
            self.crop_param = self.video_info_instance.crop_param
            scale = self.get_resize_scale(is_float=True)
            self.crop_param = [round(i * scale) for i in self.crop_param]

        # disable pad crop if using Anime4K or render based SR
        if self.is_pad_crop and self.use_sr and 'Anime4K' in self.use_sr_algo:
            self.is_pad_crop = False
            self.logger.warning("Pad Crop is not compatible with Anime4K, disabled")

    def _update_task_queue_size_by_memory(self):
        # Guess Memory and Fix Resolution
        if self.use_manual_buffer:
            # 手动指定内存占用量
            free_mem = self.manual_buffer_size * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        self.frames_queue_len = round(free_mem / (sys.getsizeof(
            np.random.rand(3, self.frame_size[0], self.frame_size[1])) / 1024 / 1024)) // 5
        if self.is_16bit_workflow:
            self.frames_queue_len = self.frames_queue_len // 3
        self.frames_queue_len = int(max(min(max(self.frames_queue_len, 8), 24), self.sr_seq_len))  # [8, 24]
        self.dup_skip_limit = int(0.5 * self.input_fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环
        self.logger.debug(f"Free RAM: {free_mem / 1024:.1f}G, "
                          f"Update Task Queue Len:  {self.frames_queue_len}, "
                          f"Duplicate Frames Cnt Upper Limit: {self.dup_skip_limit}")

    def _update_task_thread_num(self):
        return
        # if self.use_fmnet_hdr or self.use_deep_deband:
        #     if self.use_sr and self.sr_thread_cnt > 1:
        #         self.logger.warning(f"Multi-threaded SR / Multiple GPUs are not compatible with "
        #                             f"FMNet HDR or Deep Deband, "
        #                             f"set thread_cnt to 1")
        #         self.sr_thread_cnt = 1
        #     if self.use_multi_gpus:
        #         self.logger.warning(f"Multi-GPU is not compatible with "
        #                             f"FMNet HDR or Deep Deband, "
        #                             f"disable multi-gpu")
        #         self.use_multi_gpus = False

    def del_project_dir(self):
        os.chdir(appDir)  # jump out of project dir
        try:
            if os.path.exists(self.project_dir) and not os.path.samefile(self.project_dir, self.output_dir):
                shutil.rmtree(self.project_dir)
        except:
            self.logger.debug(f"Failed to remove project_dir {traceback.format_exc(limit=TB_LIMIT)}")

    def get_project_logger(self):
        return self.logger

    def update_task_info(self, update_dict: dict):
        self.task_info.update(update_dict)

    def get_main_error(self):
        if not len(self.main_error):
            return None
        else:
            return self.main_error[-1]

    def save_main_error(self, e: Exception):
        self.main_error.append(e)

    def update_preview_imgs(self, imgs: list):
        self.preview_imgs = imgs

    def get_preview_imgs(self):
        return self.preview_imgs

    def get_process_clip_points(self):
        """Get the start and end time of the input clip

        :return: start_point(int, second), end_point;
                 None as invalid
        """
        start_point, end_point = None, None
        if self.input_start_point or self.input_end_point:
            # 任意时段任务
            time_fmt = "%H:%M:%S"
            zero_point = datetime.datetime.strptime("00:00:00", time_fmt)

            if self.input_start_point is not None:
                start_point = (datetime.datetime.strptime(self.input_start_point,
                                                          time_fmt) - zero_point).total_seconds()
            if self.input_end_point is not None:
                end_point = (datetime.datetime.strptime(self.input_end_point,
                                                        time_fmt) - zero_point).total_seconds()

            if self.video_info_instance.duration > 0:
                if start_point is not None and start_point > self.video_info_instance.duration:
                    start_point = None
                    self.input_start_point = None
                    self.logger.warning("Input Start Point is larger than video duration, set to None")
                if end_point is not None and end_point > self.video_info_instance.duration:
                    end_point = None
                    self.input_end_point = None
                    self.logger.warning("Input End Point is larger than video duration, set to None")

        return start_point, end_point

    def get_resize_scale(self, is_float=False):
        scale = 0
        if all(self.resize_param) and all(self.frame_size):
            resize_resolution = self.resize_param[0] * self.resize_param[1]
            original_resolution = self.frame_size[0] * self.frame_size[1]
            scale = math.sqrt(resize_resolution / original_resolution)
            if not is_float:
                return int(math.ceil(scale))
            return scale  # could be float
        if scale <= 0:
            return RESIZE_INDEX.get_regular_sr_ratio(self.resize_settings_index)
        return 1  # invalid scale

    def is_resolution_compatible_with_vs(self):
        """
        w, h should be able to be divided by 8
        :return:
        """
        w, h = self.resize_param
        return w % 8 == 0 and h % 8 == 0


class AutoArgumentManager:
    """Manage Parameters marked as AUTO at TaskArgument startup procedure

    """

    def __init__(self, __args: TaskArgumentManager):
        self.args = __args
        self.is_modified = False

    def _modify_sr_model_argument(self):
        if self.args.release_type != RELEASE_TYPE.SAE:
            return
        if self.args.svfi_preset < 2:  # fast
            self.args.sr_tilesize = 128
            self.args.sr_realCUGAN_low_vram_mode = 3  # fast sync
        else:
            self.args.sr_tilesize = 256
            self.args.sr_realCUGAN_low_vram_mode = 0  # no sync
        self.is_modified = True

    def _modify_vfi_model_argument(self):
        if VFI_TYPE.get_model_version(self.args.vfi_model) != VFI_TYPE.AUTO or self.args.release_type == RELEASE_TYPE.SAE:
            return
        if self.args.cuda_card_cnt:  # use cuda rife model
            algo_type = "rife"  # default to RIFE
        else:  # use ncnn rife model
            algo_type = "ncnn_rife"
        if self.args.release_type in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE]:  # use ncnn_rife compulsorily
            algo_type = "ncnn_rife"

        models = VFI_TYPE.get_available_models("vfi", algo_type)
        self.args.use_rife_forward_ensemble = False
        if self.args.svfi_preset <= 2:  # fast or medium
            models = list(filter(lambda x: '4.' in x and '_trt' not in x, models))
            if self.args.svfi_preset == 0:  # fastest
                self.args.rife_scale = 0.5
        # elif self.args.svfi_preset == 2:
        #     models = list(filter(lambda x: '2.' in x, models))
        else:  # slow
            if algo_type == "ncnn_rife":
                models = list(filter(lambda x: '2.' in x, models))
            elif not self.args.is_free and self.args.cuda_card_cnt > 0 and is_cuda_ext_ok():
                algo_type = "GmfSs"
                models = VFI_TYPE.get_available_models("vfi", "GmfSs")  # override RIFE default
                models = list(filter(lambda x: '104' in x, models))  # GMFSS PG 104
            else:
                models = list(filter(lambda x: 'rpr' in x, models))  # ncnn not support rpr for now
            if self.args.svfi_preset > 3:  # veryslow
                self.args.use_rife_forward_ensemble = True
        if not models:
            model = VFI_TYPE.get_available_models("vfi", algo_type)[0]  # index issue
        else:
            models.reverse()
            model = models[0]
        self.args.vfi_algo = algo_type
        self.args.vfi_model = model
        self.is_modified = True

    def _modify_encoder_preset(self):
        presets = EncodePresetAssemply.encoder[self.args.render_encoder][
            self.args.render_encode_format]  # slow fast medium
        if self.args.svfi_preset < 2:  # fast
            self.args.render_encoder_preset = presets[1]
        elif self.args.svfi_preset == 2:  # medium
            self.args.render_encoder_preset = presets[2]
        elif self.args.svfi_preset > 2:  # slow
            self.args.render_encoder_preset = presets[0]
            if self.args.svfi_preset == 4 and 'veryslow' in presets:  # CPU veryslow
                self.args.render_encoder_preset = "veryslow"

    def _modify_encoder_format(self):
        video_info = self.args.video_info_instance.video_info
        # input_codec = video_info.get('codec_name', "hevc")
        pix_fmt = video_info.get('pix_fmt', "yuv420p10le")
        formats = EncodePresetAssemply.get_encoder_format(self.args.render_encoder, "8bit")
        # NOQA: some hardware encoders do not support 8K, like QSVENCC, but will switch to CPU automatically
        if self.args.resize_param[1] > 2160 or \
                (not self.args.render_only and self.args.target_fps > 72 and not self.args.is_render_slow_motion):
            formats = EncodePresetAssemply.get_encoder_format(self.args.render_encoder, "H265")
        if '10' in pix_fmt or self.args.video_hdr_mode != HDR_STATE.NONE:
            _formats = EncodePresetAssemply.get_encoder_format(self.args.render_encoder, "H265,10bit")
            if _formats:
                formats = _formats
        if formats:  # in case formats got failed, H264 8bit as default
            _format = formats[0]
        else:
            _format = "H264,8bit"
        self.args.render_encode_format = _format

    def _modify_encoder(self):
        if self.args.svfi_preset <= 2 and self.args.video_hdr_mode in [HDR_STATE.NONE, HDR_STATE.CUSTOM_HDR,
                                                                       HDR_STATE.NOT_CHECKED]:  # fast preset, hardware encode prioritize
            is_encc_limited = (self.args.is_free or self.args.release_type in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE] or
                               (self.args.use_sr and 'Anime4K' in self.args.use_sr_algo) or
                               self.args.is_resolution_compatible_with_vs()
                               )
            if self.args.intel_card_cnt > 0:
                if not is_encc_limited:  # is professional
                    self.args.render_encoder = "QSVENCC"
                else:
                    self.args.render_encoder = "QSV"
            elif self.args.amd_card_cnt > 0:
                if not is_encc_limited:  # is professional
                    self.args.render_encoder = "VCEENCC"
                else:
                    self.args.render_encoder = "VCE"
            elif self.args.cuda_card_cnt > 0 or self.args.release_type == RELEASE_TYPE.DEMO:  # has nvidia card
                if not is_encc_limited:  # is professional
                    # TODO Encoder DLC
                    self.args.render_encoder = "NVENCC"
                else:  # no nvidia card
                    self.args.render_encoder = "NVENC"
            else:
                self.args.render_encoder = "CPU"
        if self.args.render_encoder == "AUTO":  # no change before
            self.args.render_encoder = "CPU"

    def _modify_encoder_argument(self):
        if self.args.render_encoder == "AUTO":
            self._modify_encoder()
            self.is_modified = True
        if self.args.render_encode_format == "AUTO":
            self._modify_encoder_format()
            self.is_modified = True
        if self.args.render_encoder_preset == "AUTO":
            self._modify_encoder_preset()
            self.is_modified = True

    def modify(self):
        self._modify_encoder_argument()
        self._modify_vfi_model_argument()
        self._modify_sr_model_argument()
        if self.is_modified:
            self.args.dumb_mode = True


class OverTimeReminderTask:
    def __init__(self, interval: float, function_name, function_warning):
        self.start_time = time.time()
        self.interval = interval
        self.function_name = function_name
        self.function_warning = function_warning
        self._is_active = True

    def is_overdue(self):
        return time.time() - self.start_time > self.interval

    def is_active(self):
        return self._is_active

    def get_msgs(self):
        return self.function_name, self.interval, self.function_warning

    def deactive(self):
        self._is_active = False


def tle_classmethod_decorator(interval: int, msg_1="Function Type", msg_2="Function Warning"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            nonlocal msg_1
            if not len(msg_1) and 'name' in self.__dict__:
                msg_1 = self.__dict__['name']
            _over_time_reminder_task = OverTimeReminderTask(interval, msg_1, msg_2)
            ArgumentManager.put_overtime_task(_over_time_reminder_task)
            result = func(self, *args, **kwargs)
            _over_time_reminder_task.deactive()
            return result

        return wrapper

    return decorator


def parse_config_to_dict(path: str):
    global_config_parser = DefaultConfigParser(allow_no_value=True)
    global_config_parser.read(path, encoding='utf-8')
    global_config_parser_items = dict(global_config_parser.items("General"))
    global_args = Tools.clean_parsed_config(global_config_parser_items)
    return global_args


def get_global_task_args_manager() -> TaskArgumentManager:
    # Parse Args
    global_args_parser = argparse.ArgumentParser(prog="#### SVFI CLI tool by Jeanna ####",
                                                 description='To enhance Long video/image sequence quality')
    global_basic_parser = global_args_parser.add_argument_group(title="Basic Settings")
    global_basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                                     help="Path of input video/image sequence folder")
    global_basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="Path of config")
    global_basic_parser.add_argument("-t", '--task-id', dest='task_id', type=str, required=True,
                                     help="13-digit Task id")
    global_basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='Concat Chunk only')
    global_basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true',
                                     help='Extract input to frames Only')
    global_basic_parser.add_argument('--render-only', dest='render_only', action='store_true', help='Render only')
    global_basic_parser.add_argument("-p", '--preview', dest='is_preview', action='store_true', help='Preview Settings')
    global_pipe_parser = global_args_parser.add_argument_group(title="Pipe Settings",
                                                               description="Set the follow parameters "
                                                                           "when '-mid' is assigned, "
                                                                           "or you will encounter exceptions."
                                                                           "Output Y4M at YUV444P10")
    global_pipe_parser.add_argument("--pipe-in", dest="is_pipe_in", action="store_true", required=False,
                                    help="This enables OLS to obtain input data from stdin")
    global_pipe_parser.add_argument("--pipe-out", dest="is_pipe_out", action="store_true", required=False,
                                    help="This enables OLS to pipe output to stdout")
    global_pipe_parser.add_argument('--pipe-iw', dest='pipe_in_width', type=int, required=False, default=0,
                                    help="Width of input raw RGB, effective when --pipe-in appointed")
    global_pipe_parser.add_argument('--pipe-ih', dest='pipe_in_height', type=int, required=False, default=0,
                                    help="Height of input raw RGB, effective when --pipe-in appointed")
    global_pipe_parser.add_argument('--pipe-in-fps', dest='pipe_in_fps', type=float, required=False, default=24,
                                    help="Input stream FPS, effective when --pipe-in appointed")
    global_pipe_parser.add_argument('--pipe-in-pixfmt', dest='pipe_pix_fmt', type=str, required=False, default="rgb24",
                                    choices=("rgb24", "rgb48be", "rgb48le", "rgb48", "gbrp", "gbrp16le"),
                                    help="Pixel format of input raw RGB, effective when --pipe-in appointed")
    global_pipe_parser.add_argument('--pipe-rgb', dest='is_pipe_rgb', action='store_true',
                                    help='Pipe RGB Raw data to stdout, effective when --pipe-out appointed')
    global_pipe_parser.add_argument('--pipe-colormatrix', dest='pipe_colormatrix', type=str, required=False,
                                    default="709",
                                    choices=("470bg", "170m", "2020ncl", "709"),
                                    help="Colormatrix for RGB-YUV Conversion, effective when --pipe-in appointed, --pipe-rgb not appointed")
    # Clean Args
    global_args_read = global_args_parser.parse_args()
    if os.path.splitext(global_args_read.config)[1] == ".ini":
        global_args = parse_config_to_dict(global_args_read.config)
    else:  # json
        with open(global_args_read.config, 'r', encoding='utf-8') as r:
            global_args = json.load(r)
    global_args.update(vars(global_args_read))  # update -i -o -c and other parameters to config
    with open('current_config.json', 'w', encoding='utf-8') as w:
        json.dump(global_args, w, skipkeys=True, indent=4)
    # Read Global Task Args
    global_task_args_manager = TaskArgumentManager(global_args)
    return global_task_args_manager
