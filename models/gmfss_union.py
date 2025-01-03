# for high quality output
from models.model_gmfss_union.GMFSS import Model
from models.rife_426_heavy.IFNet_HDv3 import IFNet
from models.drm import calc_drm_gmfss, calc_drm_rife_auxiliary
from models.utils.tools import *
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
    def inference_ts_drba(self, _I0, _I1, _I2, ts, _reuse=None, linear=False):

        reuse_i1i0 = self.model.reuse(_I1, _I0, self.scale) if _reuse is None else _reuse
        reuse_i1i2 = self.model.reuse(_I1, _I2, self.scale)

        flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
        flow12, metric12 = reuse_i1i2[0], reuse_i1i2[2]

        _I0s, _I1s, _I2s = [torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                            for x in [_I0, _I1, _I2]]

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

                drm_gmfss = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, _I0s.shape[2:]) for k, v in drm_rife.items()}

                rife = self.ifnet(torch.cat((_I1s, _I0s), 1), timestep=drm_rife['drm_t1_t01'], scale_list=scale_list)[0]

                out = self.model.inference(_I1, _I0, reuse_i1i0, timestep0=drm_gmfss['drm1t_t01'],
                                           timestep1=drm_gmfss['drm0t_t01'], rife=rife)

                output.append(out)

            elif 1 < t < 2:
                t = t - 1

                drm_gmfss = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, _I0s.shape[2:]) for k, v in drm_rife.items()}

                rife = self.ifnet(torch.cat((_I1s, _I2s), 1), timestep=drm_rife['drm_t1_t12'], scale_list=scale_list)[0]

                out = self.model.inference(_I1, _I2, reuse_i1i2, timestep0=drm_gmfss['drm1t_t12'],
                                           timestep1=drm_gmfss['drm2t_t12'], rife=rife)

                output.append(out)

        # next reuse_i1i0 = reverse(current reuse_i1i2)
        # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
        # _reuse = (f1, f0, m1, m0, feat1, feat0)
        _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]

        return output, _reuse
