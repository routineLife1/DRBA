# encoding: utf-8
import time

from Utils.ArgumentsModule import ArgumentManager, TaskArgumentManager
from Utils.StaticParameters import RELEASE_TYPE

# Validation Module Initiation
if ArgumentManager.is_steam:
    from Utils.LicenseModule import SteamValidation as ValidationModule
    # Steam Initiation in MAIN
else:
    from Utils.LicenseModule import RetailValidation as ValidationModule


class ValidationFlow(ValidationModule):
    """Validate Workflow on initiation

    Sub class from either Steam or Retail

    """

    def __init__(self, _args: TaskArgumentManager):
        self.args = _args
        super().__init__(self.args.logger)
        self.kill = False

    def steam_update_achv(self, output_path):
        """Update Steam Achievement

        :return:
        """
        if not self.args.is_steam or self.kill:
            # If encountered serious error in the process, end steam update
            return
        # Get Stat
        stat_int_finished_cnt = self.GetStat("STAT_INT_FINISHED_CNT", int)
        stat_float_finished_minute = self.GetStat("STAT_FLOAT_FINISHED_MIN", float)
        stat_int_sr_finished_cnt = self.GetStat("STAT_INT_SR_FINISHED_CNT", int)
        stat_float_sr_finished_min = self.GetStat("STAT_FLOAT_SR_FINISHED_MIN", float)

        # Get ACHV
        achv_task_frozen = self.GetAchv("ACHV_Task_Frozen")
        achv_task_cruella = self.GetAchv("ACHV_Task_Cruella")
        achv_task_suzumiya = self.GetAchv("ACHV_Task_Suzumiya")
        achv_task_1000m = self.GetAchv("ACHV_Task_1000M")
        achv_task_10 = self.GetAchv("ACHV_Task_10")
        achv_task_50 = self.GetAchv("ACHV_Task_50")
        achv_sr_task_10 = self.GetAchv("ACHV_Sr_Task_10")
        achv_sr_task_50 = self.GetAchv("ACHV_Sr_Task_50")

        # Update Stat
        if self.args.is_vfi_flow_passed:
            stat_int_finished_cnt += 1
            self.SetStat("STAT_INT_FINISHED_CNT", stat_int_finished_cnt)
            if self.args.all_frames_cnt >= 0:
                # Update Mission Process Time only in interpolation
                stat_float_finished_minute += self.args.all_frames_cnt / self.args.target_fps / 60
                self.SetStat("STAT_FLOAT_FINISHED_MIN", round(stat_float_finished_minute, 2))
        if self.args.use_sr:
            stat_int_sr_finished_cnt += 1
            self.SetStat("STAT_INT_SR_FINISHED_CNT", stat_int_sr_finished_cnt)
            if self.args.all_frames_cnt >= 0:
                # Update Mission Process Time only in interpolation
                stat_float_sr_finished_min += self.args.all_frames_cnt / self.args.target_fps / 60
                self.SetStat("STAT_FLOAT_SR_FINISHED_MIN", round(stat_float_sr_finished_min, 2))

        # Update ACHV
        if 'Frozen' in output_path and not achv_task_frozen:
            self.SetAchv("ACHV_Task_Frozen")
        if 'Cruella' in output_path and not achv_task_cruella:
            self.SetAchv("ACHV_Task_Cruella")
        if any([i in output_path for i in ['Suzumiya', 'Haruhi', '涼宮', '涼宮ハルヒの憂鬱', '涼宮ハルヒの消失', '凉宫春日']]) \
                and not achv_task_suzumiya:
            self.SetAchv("ACHV_Task_Suzumiya")

        if self.args.is_vfi_flow_passed:
            if stat_int_finished_cnt > 10 and not achv_task_10:
                self.SetAchv("ACHV_Task_10")
            if stat_int_finished_cnt > 50 and not achv_task_50:
                self.SetAchv("ACHV_Task_50")
            if stat_float_finished_minute > 1000 and not achv_task_1000m:
                self.SetAchv("ACHV_Task_1000M")
        if self.args.use_sr:
            if stat_int_sr_finished_cnt > 10 and not achv_sr_task_10:
                self.SetAchv("ACHV_Sr_Task_10")
            if stat_int_sr_finished_cnt > 50 and not achv_sr_task_50:
                self.SetAchv("ACHV_Sr_Task_50")
        self.Store()

    def check_validate_functions(self, is_pro_dlc=False):
        """Check Critic functions that need purchase dlc

        raise Error if found such functions enabled

        :return:
        """
        _msg = "SVFI - Professional DLC Not Purchased,"
        # if self.args.input_start_point is not None or self.args.input_end_point is not None:
        #     raise GenericSteamException(f"{_msg} Manual Input Section Unavailable")
        # Check SAE
        if not self.args.render_only and self.args.release_type == RELEASE_TYPE.SAE:
            raise AssertionError("SAE: Frame Extraction and VFI Module Unavailable")
        # Check Demo
        if self.args.use_sr and not (self.args.release_type in [RELEASE_TYPE.DEMO, RELEASE_TYPE.SAE] or is_pro_dlc):
            # Demo/SAE is allowed to use ncnn sr
            raise AssertionError(f"{_msg} Super Resolution Module Unavailable")
        # Check Community Version
        if (self.args.is_scdet_output or self.args.is_scdet_mix) and not is_pro_dlc:
            raise AssertionError(f"{_msg} Scdet Output/Mix Unavailable")
        if self.args.use_multi_gpus and not is_pro_dlc:
            raise AssertionError(f"{_msg} Multi Video Cards Work flow Unavailable")
        if self.args.use_deinterlace and not is_pro_dlc:
            raise AssertionError(f"{_msg} DeInterlace is Unavailable")
        if self.args.use_rife_auto_scale and not is_pro_dlc:
            raise AssertionError(f"{_msg} RIFE Dynamic Scale is Unavailable")
        if self.args.is_rife_reverse and not is_pro_dlc:
            raise AssertionError(f"{_msg} RIFE Reversed Flow is Unavailable")

    def check_validation(self):
        """Check Validation Status

        raise Error if found certain check-point failed

        :return:
        """

        wait_time = 30
        while not self.CheckInit() and wait_time > 0:
            time.sleep(0.01)
            wait_time -= 0.01

        if not self.CheckValidateStart():
            if self.GetValidateError() is not None:
                self.logger.error("Validation Failed: ")
                raise self.GetValidateError()
            raise AssertionError("Validation Failed")
        is_pro_dlc = self.CheckDLC(1718750)  # Professional DLC
        self.check_validate_functions(is_pro_dlc=is_pro_dlc)
