# for study only
from models.model_gmfss.GMFSS import Model
from models.drm import calc_drm_gmfss, get_drm_t
import torch


class GMFSS:
    def __init__(self, weights=r'weights/train_log_gmfss', scale=1.0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.scale = scale
        self.pad_size = 64

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, _I0, _I1, ts):
        reuse = self.model.reuse(_I0, _I1, self.scale)

        _output = []
        for t in ts:
            if t == 0:
                _output.append(_I0)
            elif t == 1:
                _output.append(_I1)
            else:
                _output.append(
                    self.model.inference(_I0, _I1, reuse, timestep0=t, timestep1=1 - t)
                )

        return _output

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, _I0, _I1, _I2, ts, _reuse=None, linear=False):

        reuse_i1i0 = self.model.reuse(_I1, _I0, self.scale) if _reuse is None else _reuse
        reuse_i1i2 = self.model.reuse(_I1, _I2, self.scale)

        flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
        flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]

        drm01, drm10, drm12, drm21 = calc_drm_gmfss(flow10, flow12, metric10, metric12)

        output = []

        for t in ts:
            if t == 0:
                output.append(_I0)
            elif t == 1:
                output.append(_I1)
            elif t == 2:
                output.append(_I2)
            elif 0 < t < 1:
                t = 1 - t
                if linear:
                    drm1t0_unaligned = (1 - drm10) * t * 2
                    drm0t1_unaligned = 1 - drm01 * t * 2
                else:
                    drm1t0_unaligned = get_drm_t(1 - drm10, t)
                    drm0t1_unaligned = 1 - get_drm_t(drm01, t)
                # The unaligned DRM will be automatically aligned during the inference of GMFSS model.
                out = self.model.inference(_I1, _I0, reuse_i1i0, timestep0=drm1t0_unaligned,
                                           timestep1=drm0t1_unaligned)
                output.append(out)

            elif 1 < t < 2:
                t = t - 1
                if linear:
                    drm1t2_unaligned = (1 - drm12) * t * 2
                    drm2t1_unaligned = 1 - drm21 * t * 2
                else:
                    drm1t2_unaligned = get_drm_t(1 - drm12, t)
                    drm2t1_unaligned = 1 - get_drm_t(drm21, t)
                # The unaligned DRM will be automatically aligned during the inference of GMFSS model.
                out = self.model.inference(_I1, _I2, reuse_i1i2, timestep0=drm1t2_unaligned,
                                           timestep1=drm2t1_unaligned)
                output.append(out)

        # next reuse_i1i0 = reverse(current reuse_i1i2)
        # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
        # _reuse = (f1, f0, m1, m0, feat1, feat0)
        _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]

        return output, _reuse

    # Deprecated: Code below is no longer in use and may be removed in the future.
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
    #     output1, output2 = list(), list()
    #     _output = list()
    #
    #     # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
    #     if _left_scene or _right_scene:
    #         drm01, drm10, drm12, drm21 = (ones_mask.clone() * 0.5 for _ in range(4))
    #
    #     if times % 2:
    #         ts = [(i + 1) / times for i in range((times - 1) // 2)]
    #     else:
    #         ts = [(i + 0.5) / times for i in range(times // 2)]
    #
    #     for t in ts:
    #         if not _left_scene:
    #             # Adjust timestep parameters for interpolation between frames I0, I1, and I2
    #             # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2
    #             output1.append(self.model.inference(_I1, _I0, reuse_i1i0, timestep0=get_drm_t(1 - drm10, t),
    #                                                 timestep1=1 - get_drm_t(drm01, t)))
    #         else:
    #             output1.append(_I1)
    #         if not _right_scene:
    #             output2.append(self.model.inference(_I1, _I2, reuse_i1i2, timestep0=get_drm_t(1 - drm12, t),
    #                                                 timestep1=1 - get_drm_t(drm21, t)))
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
