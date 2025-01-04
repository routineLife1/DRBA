# for real-time playback(+TensorRT)
from models.rife_426_heavy.IFNet_HDv3 import IFNet
from models.drm import calc_drm_rife
from models.utils.tools import *
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

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, I0, I1, ts):
        output = []
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            else:
                output.append(
                    self.ifnet(torch.cat((I0, I1), 1), timestep=t, scale_list=scale_list)[0]
                )

        return output

    def calc_flow(self, a, b, f0=None, f1=None):
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]

        # calc flow at the lowest resolution (significantly faster with almost no quality loss).
        timestep = (a[:, :1].clone() * 0 + 1) * 0.5
        f0 = self.ifnet.encode(a[:, :3]) if f0 is None else f0
        f1 = self.ifnet.encode(b[:, :3]) if f1 is None else f1
        flow, _, _ = self.ifnet.block0(torch.cat((a[:, :3], b[:, :3], f0, f1, timestep), 1), None,
                                       scale=scale_list[0])

        # get flow flow0.5 -> 0/1
        flow50, flow51 = flow[:, :2], flow[:, 2:]

        warp_method = 'avg'

        # only need forward direction flow
        flow05_primary = warp(flow51, flow50, None, warp_method)
        flow15_primary = warp(flow50, flow51, None, warp_method)

        # qvi
        # flow05, norm2 = fwarp(flow50, flow50)
        # flow05[norm2]...
        # flow05 = -flow05

        # Another forward flow generation method may be used for filling gaps.
        flow05_secondary = -warp(flow50, flow50, None, warp_method)
        flow15_secondary = -warp(flow51, flow51, None, warp_method)

        ones_mask = flow50.clone() * 0 + 1

        warped_ones_mask_05 = warp(ones_mask, flow50, None, warp_method)
        warped_ones_mask_15 = warp(ones_mask, flow51, None, warp_method)

        holes_05 = warped_ones_mask_05 < 0.999
        holes_15 = warped_ones_mask_15 < 0.999

        flow05_primary[holes_05] = flow05_secondary[holes_05]
        flow15_primary[holes_15] = flow15_secondary[holes_15]

        flow01 = flow05_primary * 2
        flow10 = flow15_primary * 2

        return flow01, flow10, f0, f1

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, I0, I1, I2, ts, reuse=None, linear=False):

        flow10, flow01, f1, f0 = self.calc_flow(I1, I0) if not reuse else reuse
        if reuse is None:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2)
        else:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2, f0=reuse[2])

        output = list()
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            elif t == 2:
                output.append(I2)
            elif 0 < t < 1:
                t = 1 - t
                drm = calc_drm_rife(t, flow10, flow12, linear)
                inp = torch.cat((I1, I0), 1)
                out = self.ifnet(inp, timestep=drm['drm_t1_t01'], scale_list=scale_list, f0=f1, f1=f0)[0]
                output.append(out)
            elif 1 < t < 2:
                t = t - 1
                drm = calc_drm_rife(t, flow10, flow12, linear)
                inp = torch.cat((I1, I2), 1)
                out = self.ifnet(inp, timestep=drm['drm_t1_t12'], scale_list=scale_list, f0=f1, f1=f2)[0]
                output.append(out)

        # next flow10, flow01 = reverse(current flow12, flow21)
        return output, (flow21, flow12, f2, f1)
