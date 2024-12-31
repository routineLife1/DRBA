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
    def inference_ts(self, _I0, _I1, ts):
        _output = []
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        for t in ts:
            if t == 0:
                _output.append(_I0)
            elif t == 1:
                _output.append(_I1)
            else:
                _output.append(
                    self.ifnet(torch.cat((_I0, _I1), 1), timestep=t, scale_list=scale_list)[0]
                )

        return _output

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

        # Another forward flow generation method may be used for filling gaps.
        flow05_secondary = -warp(flow50, flow50, None, warp_method)
        flow15_secondary = -warp(flow51, flow51, None, warp_method)

        ones_mask = torch.ones_like(flow50, device=flow50.device)

        warped_ones_mask_05 = warp(ones_mask, flow50, None, warp_method)
        warped_ones_mask_15 = warp(ones_mask, flow51, None, warp_method)

        holes_05 = warped_ones_mask_05 < 0.999
        holes_15 = warped_ones_mask_15 < 0.999

        flow05_primary[holes_05] = flow05_secondary[holes_05]
        flow15_primary[holes_15] = flow15_secondary[holes_15]

        _flow01 = flow05_primary * 2
        _flow10 = flow15_primary * 2

        return _flow01, _flow10

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, _I0, _I1, _I2, ts, _reuse=None, linear=False):

        flow10, flow01 = self.calc_flow(_I1, _I0) if not _reuse else _reuse
        flow12, flow21 = self.calc_flow(_I1, _I2)

        output = list()
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
                drm0t1, _ = calc_drm_rife(t, flow10, flow12, linear)
                drm0t1 = resize(drm0t1, _I0.shape[2:])
                # why use drm0t1 not drm1t0, because rife use backward warp not forward warp.
                out = self.ifnet(torch.cat((_I1, _I0), 1), timestep=drm0t1, scale_list=scale_list)[0]
                output.append(out)
            elif 1 < t < 2:
                t = t - 1
                _, drm2t1 = calc_drm_rife(t, flow10, flow12, linear)
                drm2t1 = resize(drm2t1, _I0.shape[2:])
                # why use drm2t1 not drm1t2, because rife use backward warp not forward warp.
                out = self.ifnet(torch.cat((_I1, _I2), 1), timestep=drm2t1, scale_list=scale_list)[0]
                output.append(out)

        # next flow10, flow01 = reverse(current flow12, flow21)
        return output, (flow21, flow12)

    # Deprecated: Code below is no longer in use and may be removed in the future.
    # @torch.inference_mode()
    # @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    # def inference_times(self, _I0, _I1, _I2, _left_scene, _right_scene, times, _reuse=None):
    #
    #     flow10_p, flow01_p, flow01_s, flow10_s = self.calc_flow(_I1, _I0) if not _reuse else _reuse
    #     flow12_p, flow21_p, flow12_s, flow21_s = self.calc_flow(_I1, _I2)
    #
    #     ones_mask = torch.ones_like(flow10_p[:, :1], device=flow10_p.device)
    #
    #     output1, output2 = list(), list()
    #     _output = list()
    #
    #     disable_drm = False
    #     # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
    #     if _left_scene or _right_scene:
    #         drm01r, drm21r = ones_mask.clone() * 0.5, ones_mask.clone() * 0.5
    #         drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
    #         drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear', align_corners=False)
    #         disable_drm = True
    #
    #     if times % 2:
    #         ts = [(i + 1) / times for i in range((times - 1) // 2)]
    #     else:
    #         ts = [(i + 0.5) / times for i in range(times // 2)]
    #
    #     for t in ts:
    #         if not disable_drm:
    #             drm01r, drm21r = calc_drm_rife(t, flow10_p, flow12_p, flow10_s, flow12_s)
    #             drm01r = torch.nn.functional.interpolate(drm01r, size=_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #             drm21r = torch.nn.functional.interpolate(drm21r, size=_I0.shape[2:], mode='bilinear',
    #                                                      align_corners=False)
    #
    #         if not _left_scene:
    #             I10 = self.ifnet(torch.cat((_I1, _I0), 1), timestep=t * (2 * drm01r),
    #                              scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                          1 / self.scale])[0]
    #             output1.append(I10)
    #         else:
    #             output1.append(_I1)
    #
    #         if not _right_scene:
    #             I12 = self.ifnet(torch.cat((_I1, _I2), 1), timestep=t * (2 * drm21r),
    #                              scale_list=[16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale,
    #                                          1 / self.scale])[0]
    #             output2.append(I12)
    #         else:
    #             output2.append(_I1)
    #
    #     if times % 2:
    #         _output = list(reversed(output1)) + [_I1] + output2
    #     else:
    #         _output = list(reversed(output1)) + output2
    #
    #     # next flow10, flow01 = reverse(current flow12, flow21)
    #     return _output, (flow21_p, flow12_p, flow21_s, flow12_s)
