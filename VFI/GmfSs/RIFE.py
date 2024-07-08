import sys
import traceback

import torch
import torch.nn.functional as F
# from line_profiler_pycharm import profile

from Utils.StaticParameters import GLOBAL_PARAMETERS, RGB_TYPE
from Utils.utils import get_global_settings_from_local_jsons
from VFI.GmfSs.FusionNet import AnimeInterp
from train_vfss.VFI.GmfSs.MetricNet import MetricNet
from VFI.OpticalFlow.gmflow.gmflow import GMFlow
from VFI.RIFE import RIFEAnyTimeModelBase

device = torch.device(f"cuda:{GLOBAL_PARAMETERS.CURRENT_CUDA_ID}" if torch.cuda.is_available() else "cpu")


class Model(RIFEAnyTimeModelBase):
    def __init__(self, forward_ensemble=False, tta=0, ada=False, local_rank=-1):
        super().__init__(tta, forward_ensemble, ada)
        self.device_count = torch.cuda.device_count()
        self.flownet = GMFlow(num_scales=2, upsample_factor=4)
        self.metricnet = MetricNet()
        self.fusionnet = AnimeInterp()
        self.version = 7.0

        global_settings = get_global_settings_from_local_jsons()
        self.is_graph_enabled = global_settings.get('is_cuda_graph_enabled', False)
        self.is_graph_enabled = self.is_graph_enabled and not self.ada and torch.cuda.is_available()
        self.flow_graph: torch.cuda.CUDAGraph = None
        self.is_flow_graph_initialized = False
        self.static_flow_img0: torch.Tensor = None
        self.static_flow_img1: torch.Tensor = None
        self.static_flow01: torch.Tensor = None
        self.static_flow10: torch.Tensor = None

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
            self.metricnet.load_state_dict(convert(torch.load('{}/metric.pkl'.format(path))))
            self.fusionnet.load_state_dict(convert(torch.load('{}/fusionnet.pkl'.format(path))))

    def inference(self, img0, img1, flow01, flow10, metric0, metric1, feat1_pyramid, feat2_pyramid, timestep=0.5):

        F0t = timestep * flow01
        F1t = (1 - timestep) * flow10

        out = self.fusionnet(img0, img1, F0t, F1t, metric0, metric1, feat1_pyramid, feat2_pyramid)

        return torch.clamp(out, 0, 1)

    def _gmfss_feature_extract(self, img0, img1):
        """Feature Extraction for Different deviations of GmfSs, for reuse

        :param img0: Tensor of BCHW
        :param img1: Tensor of BCHW
        :return:
        """
        feat1_pyramid = self.fusionnet.feat_ext((img0 - 0.5) / 0.5)
        feat2_pyramid = self.fusionnet.feat_ext((img1 - 0.5) / 0.5)
        return feat1_pyramid, feat2_pyramid

    def _gmfss_preprocess(self, img):
        """Preprocess of input tensor for different deviations of GmfSs, for (inherit).

        :param img:
        :return:
        """
        return img

    # @profile
    def get_inferences(self, img0, img1, n=1, ts: list = None, scale=1.0, h=0, w=0):
        from torch.nn import functional as F
        inferenced = list()
        t = ts if ts is not None and len(ts) else self.get_anytime_t(n)

        # Extract Feature Pyramids first, and reuse
        feat1_pyramid, feat2_pyramid = self._gmfss_feature_extract(img0, img1)

        # Preprocess
        img0, img1 = self._gmfss_preprocess(img0), self._gmfss_preprocess(img1)
        # torch.cuda.synchronize()

        # Reuse Flows and Metrics
        flow01, flow10 = self.get_flow(img0, img1, scale)
        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)
        # torch.cuda.synchronize()

        for frame_index, t_value in enumerate(t):
            prediction = self.inference(img0, img1, flow01, flow10, metric0, metric1, feat1_pyramid, feat2_pyramid,
                                        t_value)
            # torch.cuda.synchronize()

            prediction = F.interpolate(prediction, (h, w), mode='bilinear', align_corners=False)
            prediction = torch.clamp(prediction.float(), 0, 1) * RGB_TYPE.SIZE
            prediction = prediction.cpu()
            inferenced.append(prediction)
            del prediction
        return inferenced

    # @profile
    def get_flow(self, img0, img1, scale):
        """
        :param img0:
        :param img1:
        :param scale:
        :return:
        """
        def _get_flow(_img0, _img1):
            pred_bidir_flow = self.forward_ensemble
            if scale != 1.0:
                imgf0 = F.interpolate(_img0, scale_factor=scale, mode="bilinear", align_corners=False)
                imgf1 = F.interpolate(_img1, scale_factor=scale, mode="bilinear", align_corners=False)
            else:
                imgf0, imgf1 = _img0, _img1
            if not pred_bidir_flow:
                _flow01 = self.flownet(imgf0, imgf1, pred_bidir_flow=pred_bidir_flow)
                _flow10 = self.flownet(imgf1, imgf0, pred_bidir_flow=pred_bidir_flow)
            else:
                flow = self.flownet(imgf0, imgf1, pred_bidir_flow=pred_bidir_flow)
                _flow01 = flow[0].unsqueeze(0)
                _flow10 = flow[1].unsqueeze(0)
            if scale != 1.0:
                _flow01 = F.interpolate(_flow01, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
                _flow10 = F.interpolate(_flow10, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
            return _flow01, _flow10

        def _initiate_flow_graph(_img0, _img1):
            self.static_flow_img0 = _img0
            self.static_flow_img1 = _img1

            torch.cuda.synchronize(device=device)
            _get_flow(_img0, _img1)
            torch.cuda.synchronize(device=device)

            self.flow_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.flow_graph):
                self.static_flow01, self.static_flow10 = _get_flow(_img0, _img1)
            self.is_flow_graph_initialized = True
            return self.static_flow01, self.static_flow10
        if self.is_graph_enabled:
            if not self.is_flow_graph_initialized:
                try:
                    _initiate_flow_graph(img0, img1)
                    print(f"VFI INFO: CUDA Flow Graph initiated", file=sys.stderr)
                except RuntimeError:
                    print(f"VFI WARNING: CUDA Flow Graph is not supported on this device.\n{traceback.format_exc()}",
                          file=sys.stderr)
                    self.is_graph_enabled = False
                    return _get_flow(img0, img1)
            self.static_flow_img0.copy_(img0)
            self.static_flow_img1.copy_(img1)
            self.flow_graph.replay()
            flow01, flow10 = self.static_flow01, self.static_flow10
        else:
            flow01, flow10 = _get_flow(img0, img1)
        return flow01, flow10
