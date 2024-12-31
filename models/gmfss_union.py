# for high quality output
from models.model_gmfss_union.GMFSS import Model
from models.rife_426_heavy.IFNet_HDv3 import IFNet
from models.drm import calc_drm_gmfss, calc_drm_rife_auxiliary, get_drm_t
from models.utils.tools import *
import numpy as np
import torch
import os


class GMFSS_UNION:
    def __init__(self, weights='weights/train_log_gmfss_union',
                 scale=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.ifnet = IFNet().to(device).eval()
        self.ifnet.load_state_dict(
            convert(torch.load(os.path.join(weights, "rife.pkl"), map_location='cpu')),
            strict=False)
        self.scale = scale
        self.pad_size = 128

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, _I0, _I1, ts):
        reuse = self.model.reuse(_I0, _I1, self.scale)
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        _output = []
        for t in ts:
            if t == 0:
                _output.append(_I0)
            elif t == 1:
                _output.append(_I1)
            else:
                _I0s = F.interpolate(_I0, scale_factor=0.5, mode='bilinear', align_corners=False)
                _I1s = F.interpolate(_I1, scale_factor=0.5, mode='bilinear', align_corners=False)
                rife = self.ifnet(torch.cat((_I0s, _I1s), 1), timestep=t, scale_list=scale_list)[0]
                _output.append(
                    self.model.inference(_I0, _I1, reuse, timestep0=t, timestep1=1 - t, rife=rife)
                )

        return _output

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, _I0, _I1, _I2, ts, _reuse=None):

        reuse_i1i0 = self.model.reuse(_I1, _I0, self.scale) if _reuse is None else _reuse
        reuse_i1i2 = self.model.reuse(_I1, _I2, self.scale)

        flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
        flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]

        drm01, drm10, drm12, drm21 = calc_drm_gmfss(flow10, flow12, metric10, metric12)

        _I0s, _I1s, _I2s = [torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                            for x
                            in [_I0, _I1, _I2]]

        output = []
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]

        for t in ts:
            if t == 0:
                output.append(_I0)
            elif t == 1:
                output.append(_I1)
            elif t == 2:
                output.append(_I2)
            elif 0 < t < 1:
                t = 1 - t
                drm1t0_unaligned = get_drm_t(drm10, t)
                drm0t1_unaligned = get_drm_t(drm01, t)
                drm1t0, _ = calc_drm_rife_auxiliary(t, drm10, drm12, flow10, flow12, metric10, metric12)
                drm1t0 = resize(drm1t0, _I0s.shape[2:])

                rife = self.ifnet(torch.cat((_I1s, _I0s), 1), timestep=drm1t0, scale_list=scale_list)[0]
                # The unaligned DRM will be automatically aligned during the inference of GMFSS model.
                out = self.model.inference(_I1, _I0, reuse_i1i0, timestep0=drm1t0_unaligned,
                                           timestep1=drm0t1_unaligned, rife=rife)

                output.append(out)

            elif 1 < t < 2:
                t = t - 1
                drm1t2_unaligned = get_drm_t(drm12, t)
                drm2t1_unaligned = get_drm_t(drm21, t)
                _, drm1t2 = calc_drm_rife_auxiliary(t, drm10, drm12, flow10, flow12, metric10, metric12)
                drm1t2 = resize(drm1t2, _I0s.shape[2:])

                rife = self.ifnet(torch.cat((_I1s, _I2s), 1), timestep=drm1t2, scale_list=scale_list)[0]
                # The unaligned DRM will be automatically aligned during the inference of GMFSS model.
                out = self.model.inference(_I1, _I2, reuse_i1i2, timestep0=drm1t2_unaligned, timestep1=drm2t1_unaligned,
                                           rife=rife)

                output.append(out)

        # next reuse_i1i0 = reverse(current reuse_i1i2)
        # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
        # _reuse = (f1, f0, m1, m0, feat1, feat0)
        _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]

        return output, _reuse

    # @torch.inference_mode()
    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    # def inference_times(self, _I0, _I1, _I2, _left_scene, _right_scene, times, _reuse=None):
    #     reuse_i1i0 = self.model.reuse(_I1, _I0, self.scale) if _reuse is None else _reuse
    #     reuse_i1i2 = self.model.reuse(_I1, _I2, self.scale)
    #
    #     flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
    #     flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]
    #
    #     drm01, drm10, drm12, drm21 = calc_drm_gmfss(flow10, flow12, metric10, metric12)
    #     ones_mask = torch.ones_like(drm01, device=drm01.device)
    #
    #     f_I0, f_I1, f_I2 = [torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    #                         for x
    #                         in [_I0, _I1, _I2]]
    #
    #     output1, output2 = list(), list()
    #     _output = list()
    #
    #     disable_drm = False
    #     # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
    #     if _left_scene or _right_scene:
    #         drm01, drm10, drm12, drm21 = (ones_mask.clone() * 0.5 for _ in range(4))
    #         drm01r, drm21r = (ones_mask.clone() * 0.5 for _ in range(2))
    #         drm01r = torch.nn.functional.interpolate(drm01r, size=f_I0.shape[2:], mode='bilinear', align_corners=False)
    #         drm21r = torch.nn.functional.interpolate(drm21r, size=f_I0.shape[2:], mode='bilinear', align_corners=False)
    #         disable_drm = True
    #
    #     if times % 2:
    #         ts = [(i + 1) / times for i in range((times - 1) // 2)]
    #     else:
    #         ts = [(i + 0.5) / times for i in range(times // 2)]
    #
    #     for t in ts:
    #         # Adjust timestep parameters for interpolation between frames I0, I1, and I2
    #         # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2
    #
    #         if not disable_drm:
    #             drm01r, drm21r = calc_drm_rife_auxiliary(t, drm10, drm12, flow10, flow12, metric10, metric12)
    #             drm01r = torch.nn.functional.interpolate(drm01r, size=f_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #             drm21r = torch.nn.functional.interpolate(drm21r, size=f_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #
    #         if not _left_scene:
    #             I10 = self.ifnet(torch.cat((f_I1, f_I0), 1), timestep=t * (2 * drm01r),
    #                              scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                          1 / self.scale])[0]
    #             output1.append(self.model.inference(_I1, _I0, reuse_i1i0, timestep0=t * (2 * (1 - drm10)),
    #                                                 timestep1=1 - t * (2 * drm01), rife=I10))
    #         else:
    #             output1.append(_I1)
    #
    #         if not _right_scene:
    #             I12 = self.ifnet(torch.cat((f_I1, f_I2), 1), timestep=t * (2 * drm21r),
    #                              scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                          1 / self.scale])[0]
    #             output2.append(self.model.inference(_I1, _I2, reuse_i1i2, timestep0=t * (2 * (1 - drm12)),
    #                                                 timestep1=1 - t * (2 * drm21), rife=I12))
    #         else:
    #             output2.append(_I1)
    #
    #     if times % 2:
    #         _output = list(reversed(output1)) + [_I1] + output2
    #     else:
    #         _output = list(reversed(output1)) + output2
    #
    #     # next reuse_i1i0 = reverse(current reuse_i1i2)
    #     # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
    #     # _reuse = (f1, f0, m1, m0, feat1, feat0)
    #     _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]
    #
    #     return _output, _reuse

    # @torch.inference_mode()
    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    # def inference_ts_drba(self, _I0, _I1, _I2, minus_t, zero_t, plus_t, _left_scene, _right_scene, _reuse=None):
    #     reuse_i1i0 = self.model.reuse(_I1, _I0, self.scale) if _reuse is None else _reuse
    #     reuse_i1i2 = self.model.reuse(_I1, _I2, self.scale)
    #
    #     flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
    #     flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]
    #
    #     drm01, drm10, drm12, drm21 = calc_drm_gmfss(flow10, flow12, metric10, metric12)
    #     ones_mask = torch.ones_like(drm01, device=drm01.device)
    #
    #     f_I0, f_I1, f_I2 = [torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    #                         for x
    #                         in [_I0, _I1, _I2]]
    #
    #     output1, output2 = list(), list()
    #
    #     # The output every three inputs (I0, I1, I2) range between I0.5 and I1.5.
    #     # Therefore, when a transition occurs, the only frame can be copied is I1.
    #     if _left_scene:
    #         for _ in minus_t:
    #             zero_t = np.append(zero_t, 0)
    #         minus_t = list()
    #
    #     if _right_scene:
    #         for _ in plus_t:
    #             zero_t = np.append(zero_t, 0)
    #         plus_t = list()
    #
    #     disable_drm = False
    #     # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
    #     if _left_scene or _right_scene:
    #         drm01, drm10, drm12, drm21 = (ones_mask.clone() * 0.5 for _ in range(4))
    #         drm01r, drm21r = (ones_mask.clone() * 0.5 for _ in range(2))
    #         drm01r = torch.nn.functional.interpolate(drm01r, size=f_I0.shape[2:], mode='bilinear', align_corners=False)
    #         drm21r = torch.nn.functional.interpolate(drm21r, size=f_I0.shape[2:], mode='bilinear', align_corners=False)
    #         disable_drm = True
    #
    #     for t in minus_t:
    #         t = -t
    #         if not disable_drm:
    #             drm01r, _ = calc_drm_rife_auxiliary(t, drm10, drm12, flow10, flow12, metric10, metric12)
    #             drm01r = torch.nn.functional.interpolate(drm01r, size=f_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #
    #         I10 = self.ifnet(torch.cat((f_I1, f_I0), 1), timestep=t * (2 * drm01r),
    #                          scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                      1 / self.scale])[0]
    #         output1.append(self.model.inference(_I1, _I0, reuse_i1i0, timestep0=t * (2 * (1 - drm10)),
    #                                             timestep1=1 - t * (2 * drm01), rife=I10))
    #     for _ in zero_t:
    #         output1.append(_I1)
    #
    #     for t in plus_t:
    #         if not disable_drm:
    #             _, drm21r = calc_drm_rife_auxiliary(t, drm10, drm12, flow10, flow12, metric10, metric12)
    #             drm21r = torch.nn.functional.interpolate(drm21r, size=f_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #
    #         I12 = self.ifnet(torch.cat((f_I1, f_I2), 1), timestep=t * (2 * drm21r),
    #                          scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                      1 / self.scale])[0]
    #         output2.append(self.model.inference(_I1, _I2, reuse_i1i2, timestep0=t * (2 * (1 - drm12)),
    #                                             timestep1=1 - t * (2 * drm21), rife=I12))
    #
    #     _output = output1 + output2
    #
    #     # next reuse_i1i0 = reverse(current reuse_i1i2)
    #     # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
    #     # _reuse = (f1, f0, m1, m0, feat1, feat0)
    #     _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]
    #
    #     return _output, _reuse
