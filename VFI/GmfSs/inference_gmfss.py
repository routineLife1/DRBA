import warnings

from Utils.ArgumentsModule import ArgumentManager
from Utils.StaticParameters import VFI_TYPE
from VFI.RIFE.inference_rife import RifeMultiInterpolation

warnings.filterwarnings("ignore")
# from line_profiler_pycharm import profile
"""TEST"""


class GmfSsInterpolation(RifeMultiInterpolation):
    def __init__(self, __args: ArgumentManager):
        super().__init__(__args)
        self.initiated = False
        self.model_version = VFI_TYPE.GmfSs  # default

    def _generate_padding(self, img, scale: float):
        """

        :param scale:
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        # if '_union' in self.model_path:  # debug
        #     scale /= 2.0
        tmp = max(64, int(64 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw, 0, ph)
        return padding, h, w

    def initiate_algorithm(self):
        if self.initiated:
            return
        self._initiate_torch()
        self._check_model_path()

        self.logger.info("Loading GMF+SoftSplat Model: https://github.com/hyw-dev/GMFSS")
        self.model_version = VFI_TYPE.get_model_version(self.model_path)

        if '_up' in self.model_path:
            from VFI.GmfSs_up.RIFE import Model
        elif '_primaris' in self.model_path:
            from VFI.GmfSs_primaris.RIFE import Model
        elif '_union' in self.model_path:
            if '_trt' in self.model_path:
                from VFI.GmfSs_union.RIFE_trt import Model
            else:
                from VFI.GmfSs_union.RIFE import Model
        elif '_pg_104' in self.model_path or '_pg_108' in self.model_path:
            if '_torch_trt' in self.model_path:
                from VFI_uc.GmfSs_pg_104_trt.RIFE_trt import Model
            elif '_ts' in self.model_path:
                from VFI_uc.Gmfss_pg_104_ts.RIFE import Model
            else:
                from VFI.GmfSs_pg_104.RIFE import Model
        elif '_pg_105_quantitized' in self.model_path:
            from VFI_uc.GmfSs_pg_105_quantized.RIFE import Model
        elif '_pg_110' in self.model_path:
            from VFI.GmfSs_pg_105.RIFE import Model
        elif '_pg_109' in self.model_path:
            from VFI.GmfSs_pg_109.RIFE import Model
        elif '_real' in self.model_path:
            from VFI.GmfSs_real.RIFE import Model
        else:
            from VFI.GmfSs.RIFE import Model

        model = Model(forward_ensemble=self.args.use_rife_forward_ensemble,
                      tta=self.tta_mode, ada=self.use_auto_scale)
        model.load_model(self.model_path, -1)
        self.logger.info("GMF+SS model loaded.")

        self.model = model
        self.model.device()
        self.model.eval()

        self._print_card_info()
        self.initiated = True


if __name__ == "__main__":
    """
        Testbench has been moved to `test` folder 
        The following is for profiling
    """
    # from Utils.utils import Tools
    # from time import time
    # from tests.image import output_root, get_images
    # import cv2
    # import os
    # import numpy as np
    # from Utils.StaticParameters import VFI_TYPE, appDir, abs_path_to_vspipe_trt
    # import sys
    #
    # os.chdir(appDir)
    # os.environ['PATH'] = abs_path_to_vspipe_trt + os.pathsep + os.environ['PATH']
    # sys.path.append(appDir)
    # sys.path.append(abs_path_to_vspipe_trt)
    #
    # img0, img1 = get_images(2)
    # algo, model, scale, n = "GmfSs", "GmfSs_union_v_torch_trt", 1.0, 1
    # # algo, model, scale, n = "GmfSs", "GmfSs_union_v", 1.0, 1
    #
    # _manager = ArgumentManager({'vfi_algo': algo, 'vfi_model': model, 'rife_scale': scale})
    # _manager.logger = Tools.get_logger('test', '')
    # _inference_module = GmfSsInterpolation(_manager)
    # _inference_module.initiate_algorithm()
    # ts = []
    #
    # # flow test
    # # i0 = torch.rand(1, 3, 576, 960).cuda()
    # # i1 = torch.rand(1, 3, 576, 960).cuda()
    #
    # for _j in range(1):
    #     t = time()
    #     # with torch.no_grad():
    #     for _i, _o in enumerate(_inference_module.generate_n_interp(img0, img1, n, scale)):
    #         # t = time()
    #         # _inference_module.model.get_flow(i0, i1, 0.5)
    #         cv2.imwrite(os.path.join(output_root, f"out_{_i}_{model}_{scale}_{n}.png"), _o)
    #         pass
    #     ts.append(time() - t)
    #     print(f"Elapsed time: {ts[-1]}s")
    #
    # print(np.mean(ts[1:]))
    # print(np.var(ts[1:]))
