# for study only
from models.model_gmfss.GMFSS import Model
from models.drm import calc_drm_gmfss
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
                drm = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                out = self.model.inference(_I1, _I0, reuse_i1i0, timestep0=drm['drm1t_t01'],
                                           timestep1=drm['drm0t_t01'])
                output.append(out)

            elif 1 < t < 2:
                t = t - 1
                drm = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                out = self.model.inference(_I1, _I2, reuse_i1i2, timestep0=drm['drm1t_t12'],
                                           timestep1=drm['drm2t_t12'])
                output.append(out)

        # next reuse_i1i0 = reverse(current reuse_i1i2)
        # f0, f1, m0, m1, feat0, feat1 = reuse_i1i2
        # _reuse = (f1, f0, m1, m0, feat1, feat0)
        _reuse = [value for pair in zip(reuse_i1i2[1::2], reuse_i1i2[0::2]) for value in pair]

        return output, _reuse
