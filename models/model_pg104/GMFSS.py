import torch
import torch.nn.functional as F

from models.gmflow.gmflow import GMFlow
from models.model_pg104.MetricNet import MetricNet
from models.model_pg104.FeatureNet import FeatureNet
from models.model_pg104.FusionNet import GridNet
from models.model_pg104.softsplat import softsplat as warp

device = torch.device("cuda")


class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet(9, 64 * 2, 128 * 2, 192 * 2, 3)
        self.version = 3.9

    def half(self):
        self.flownet.half()
        self.metricnet.half()
        self.feat_ext.half()
        self.fusionnet.half()

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path)))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)))

    def reuse(self, img0, img1, scale):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
            flow10 = F.interpolate(flow10, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def inference(self, img0, img1, reuse_things, timestep):
        flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][
            1], reuse_things[4][2]
        flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][
            1], reuse_things[5][2]

        F1t = timestep * flow01
        F2t = (1 - timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1 - timestep) * metric1

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        I1t = warp(img0, F1t, Z1t, strMode='soft')
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        I2t = warp(img1, F2t, Z2t, strMode='soft')

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        out = self.fusionnet(torch.cat([I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                             torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return torch.clamp(out, 0, 1)

    def inference_t2(self, img0, img1, reuse_things, timestep0, timestep1, rife):
        flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][
            1], reuse_things[4][2]
        flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][
            1], reuse_things[5][2]

        timestep0 = torch.clamp(timestep0, 0, 1)
        timestep1 = torch.clamp(timestep1, 0, 1)

        F1t = timestep0 * flow01
        F2t = timestep1 * flow10

        Z1t = timestep0 * metric0
        Z2t = timestep1 * metric1

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        I1t = warp(img0, F1t, Z1t, strMode='soft')
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        I2t = warp(img1, F2t, Z2t, strMode='soft')

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        out = self.fusionnet(torch.cat([I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                             torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        # visualization experiment
        # import cv2
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\I1t.png", I1t[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\I2t.png", I2t[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # wtimestep0 = warp(timestep0, F1t, Z1t, strMode='soft')
        # wtimestep1 = warp(timestep1, F2t, Z2t, strMode='soft')
        #
        # ones_mask0 = warp(torch.ones_like(timestep0).to(timestep0.device), F1t, Z1t, strMode='soft') < 0.999
        # ones_mask1 = warp(torch.ones_like(timestep1).to(timestep1.device), F2t, Z2t, strMode='soft') < 0.999
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\omask0.png", ones_mask0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\omask1.png", ones_mask1[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wot0.png", wtimestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wot1.png", wtimestep1[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # wtimestep0[ones_mask0] = 1
        # wtimestep1[ones_mask1] = 1
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\t0.png", timestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\t1.png", timestep1[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wt0.png", wtimestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wt1.png", wtimestep1[0].permute(1, 2, 0).cpu().numpy() * 255)

        return torch.clamp(out, 0, 1)
