# for real-time playback(+TensorRT)
from models.rife_426_heavy.IFNet_HDv3 import IFNet
from models.drm import calc_drm_rife, calc_drm_rife_v2
from models.utils.tools import *
import numpy as np
import torch
import os

if check_cupy_env():
    from models.softsplat.softsplat import softsplat as warp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from models.softsplat.softsplat_torch import softsplat as warp


class RIFE:
    def __init__(self, weights='weights/train_log_rife_426_heavy', scale=1.0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.ifnet = IFNet().to(device).eval()
        self.ifnet.load_state_dict(convert(torch.load(os.path.join(weights, 'flownet.pkl'), map_location='cpu')),
                                   strict=False)
        self.scale = scale
        self.pad_size = 64

    def calc_flow(self, a, b):
        imgs = torch.cat((a, b), 1)
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        # get top scale flow flow0.5 -> 0/1
        flow = self.ifnet(imgs, timestep=0.5, scale_list=scale_list)[1][-1]
        flow50, flow51 = flow[:, :2], flow[:, 2:]

        warp_method = 'avg'

        # only need forward direction flow
        flow05_primary = warp(flow51, flow50, None, warp_method)
        flow15_primary = warp(flow50, flow51, None, warp_method)

        # qvi
        # flow05, norm2 = fwarp(flow50, flow50)
        # flow05[norm2]...
        # flow05 = -flow05

        flow05_secondary = -warp(flow50, flow50, None, warp_method)
        flow15_secondary = -warp(flow51, flow51, None, warp_method)

        _flow01_primary = flow05_primary * 2
        _flow10_primary = flow15_primary * 2

        _flow01_secondary = flow05_secondary * 2
        _flow10_secondary = flow15_secondary * 2

        return _flow01_primary, _flow10_primary, _flow01_secondary, _flow10_secondary

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_times(self, _I0, _I1, _I2, _left_scene, _right_scene, times, _reuse=None):

        flow10_p, flow01_p, flow01_s, flow10_s = self.calc_flow(_I1, _I0) if not _reuse else _reuse
        flow12_p, flow21_p, flow12_s, flow21_s = self.calc_flow(_I1, _I2)

        ones_mask = torch.ones_like(flow10_p[:, :1], device=flow10_p.device)

        output1, output2 = list(), list()
        _output = list()

        disable_drm = False
        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if _left_scene or _right_scene:
            drm01r, drm21r = ones_mask.clone() * 0.5, ones_mask.clone() * 0.5
            drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            disable_drm = True

        if times % 2:
            for i in range((times - 1) // 2):
                t = (i + 1) / times
                # Adjust timestep parameters for interpolation between frames I0, I1, and I2
                # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2

                if not disable_drm:
                    drm01r, drm21r = calc_drm_rife(t, flow10_p, flow12_p, flow10_s, flow12_s)
                    drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear',
                                                             align_corners=False)
                    drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear',
                                                             align_corners=False)

                if not _left_scene:
                    I10 = self.ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
                                     scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                 1 / self.scale])[0]
                    output1.append(I10)
                else:
                    output1.append(_I1)

                if not _right_scene:
                    I12 = self.ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
                                     scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                 1 / self.scale])[0]
                    output2.append(I12)
                else:
                    output2.append(_I1)

            _output = list(reversed(output1)) + [_I1] + output2
        else:
            for i in range(times // 2):
                t = (i + 0.5) / times

                if not disable_drm:
                    drm01r, drm21r = calc_drm_rife(t, flow10_p, flow12_p, flow10_s, flow12_s)
                    drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear',
                                                             align_corners=False)
                    drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear',
                                                             align_corners=False)

                if not _left_scene:
                    I10 = self.ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
                                     scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                 1 / self.scale])[0]
                    output1.append(I10)
                else:
                    output1.append(_I1)

                if not _right_scene:
                    I12 = self.ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
                                     scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                 1 / self.scale])[0]
                    output2.append(I12)
                else:
                    output2.append(_I1)

            _output = list(reversed(output1)) + output2

        # next flow10, flow01 = reverse(current flow12, flow21)
        return _output, (flow21_p, flow12_p, flow21_s, flow12_s)

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_timestamps(self, _I0, _I1, _I2, minus_t, zero_t, plus_t, _left_scene, _right_scene, _reuse=None):

        flow10_p, flow01_p, flow01_s, flow10_s = self.calc_flow(_I1, _I0) if not _reuse else _reuse
        flow12_p, flow21_p, flow12_s, flow21_s = self.calc_flow(_I1, _I2)

        ones_mask = torch.ones_like(flow10_p[:, :1], device=flow10_p.device)

        output1, output2 = list(), list()

        # The output every three inputs (I0, I1, I2) range between I0.5 and I1.5.
        # Therefore, when a transition occurs, the only frame can be copied is I1.
        if _left_scene:
            for _ in minus_t:
                zero_t = np.append(zero_t, 0)
            minus_t = list()

        if _right_scene:
            for _ in plus_t:
                zero_t = np.append(zero_t, 0)
            plus_t = list()

        disable_drm = False
        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if _left_scene or _right_scene:
            drm01r, drm21r = ones_mask.clone() * 0.5, ones_mask.clone() * 0.5
            drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            disable_drm = True

        for t in minus_t:
            t = -t
            if not disable_drm:
                drm01r, _ = calc_drm_rife(t, flow10_p, flow12_p, flow10_s, flow12_s)
                drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear',
                                                         align_corners=False)
            output1.append(self.ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
                                      scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                  1 / self.scale])[0])
        for _ in zero_t:
            output1.append(_I1)

        for t in plus_t:
            if not disable_drm:
                _, drm21r = calc_drm_rife(t, flow10_p, flow12_p, flow10_s, flow12_s)
                drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear',
                                                         align_corners=False)
            output2.append(self.ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
                                      scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                  1 / self.scale])[0])

        _output = output1 + output2

        # next flow10, flow01 = reverse(current flow12, flow21)
        return _output, (flow21_p, flow12_p, flow21_s, flow12_s)

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, _I0, _I1, ts):
        _output = []
        for t in ts:
            _output.append(
                self.ifnet(torch.cat((_I0, _I1), 1), timestep=t,
                           scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                       1 / self.scale])[0]
            )

        return _output

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_timestamps_v2(self, _I0, _I1, _I2, minus_t, zero_t, plus_t, _left_scene, _right_scene):

        flow10_p, flow01_p, flow01_s, flow10_s = self.calc_flow(_I1, _I0)
        flow12_p, flow21_p, flow12_s, flow21_s = self.calc_flow(_I1, _I2)

        ones_mask = torch.ones_like(flow10_p[:, :1], device=flow10_p.device)

        output1, output2 = list(), list()

        if _left_scene:
            for i in range(len(minus_t)):
                minus_t[i] = -1

        if _right_scene:
            for _ in plus_t:
                zero_t = np.append(zero_t, 0)
            plus_t = list()

        disable_drm = False
        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if _left_scene or _right_scene:
            drm01r, drm21r = ones_mask.clone() * 0.5, ones_mask.clone() * 0.5
            drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
            disable_drm = True

        for t in minus_t:
            t = -t
            if t == 1:
                output1.append(_I0)
                continue
            if not disable_drm:
                drm01r, _ = calc_drm_rife_v2(t, flow10_p, flow12_p, flow10_s, flow12_s)
                drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear',
                                                         align_corners=False)
            output1.append(self.ifnet(torch.cat((_I1, _I0), 1), timestep=drm01r,
                                      scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                  1 / self.scale])[0])

        for _ in zero_t:
            output1.append(_I1)

        for t in plus_t:
            if t == 1:
                # Following the principle of left-closed, right-open, the logic of this line of code will not be triggered.
                assert True
                # output2.append(_I2)
                # continue
            if not disable_drm:
                _, drm21r = calc_drm_rife_v2(t, flow10_p, flow12_p, flow10_s, flow12_s)
                drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear',
                                                         align_corners=False)
            output2.append(self.ifnet(torch.cat((_I1, _I2), 1), timestep=drm21r,
                                      scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
                                                  1 / self.scale])[0])

        _output = output1 + output2

        return _output
