import torch
import torch.nn as nn

from Utils.StaticParameters import GLOBAL_PARAMETERS
from VFI_uc.sswarp import warp, dmetric, dflow, norm_process

device = torch.device(f"cuda:{GLOBAL_PARAMETERS.CURRENT_CUDA_ID}" if torch.cuda.is_available() else "cpu")

# Residual Block
def ResidualBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    )


# downsample block
def DownsampleBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )


# upsample block
def UpsampleBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
    )


# grid network
class GridNet(nn.Module):
    def __init__(self, in_channels, in_channels1, in_channels2, in_channels3, out_channels):
        super(GridNet, self).__init__()

        self.residual_model_head = ResidualBlock(in_channels, 32, stride=1)
        self.residual_model_head1 = ResidualBlock(in_channels1, 32, stride=1)
        self.residual_model_head2 = ResidualBlock(in_channels2, 64, stride=1)
        self.residual_model_head3 = ResidualBlock(in_channels3, 96, stride=1)

        self.residual_model_01 = ResidualBlock(32, 32, stride=1)
        # self.residual_model_02=ResidualBlock(32, 32, stride=1)
        # self.residual_model_03=ResidualBlock(32, 32, stride=1)
        self.residual_model_04 = ResidualBlock(32, 32, stride=1)
        self.residual_model_05 = ResidualBlock(32, 32, stride=1)
        self.residual_model_tail = ResidualBlock(32, out_channels, stride=1)

        self.residual_model_11 = ResidualBlock(64, 64, stride=1)
        # self.residual_model_12=ResidualBlock(64, 64, stride=1)
        # self.residual_model_13=ResidualBlock(64, 64, stride=1)
        self.residual_model_14 = ResidualBlock(64, 64, stride=1)
        self.residual_model_15 = ResidualBlock(64, 64, stride=1)

        self.residual_model_21 = ResidualBlock(96, 96, stride=1)
        # self.residual_model_22=ResidualBlock(96, 96, stride=1)
        # self.residual_model_23=ResidualBlock(96, 96, stride=1)
        self.residual_model_24 = ResidualBlock(96, 96, stride=1)
        self.residual_model_25 = ResidualBlock(96, 96, stride=1)

        #

        self.downsample_model_10 = DownsampleBlock(32, 64, stride=2)
        self.downsample_model_20 = DownsampleBlock(64, 96, stride=2)

        self.downsample_model_11 = DownsampleBlock(32, 64, stride=2)
        self.downsample_model_21 = DownsampleBlock(64, 96, stride=2)

        # self.downsample_model_12=DownsampleBlock(32, 64, stride=2)
        # self.downsample_model_22=DownsampleBlock(64, 96, stride=2)

        #

        # self.upsample_model_03=UpsampleBlock(64, 32, stride=1)
        # self.upsample_model_13=UpsampleBlock(96, 64, stride=1)

        self.upsample_model_04 = UpsampleBlock(64, 32, stride=1)
        self.upsample_model_14 = UpsampleBlock(96, 64, stride=1)

        self.upsample_model_05 = UpsampleBlock(64, 32, stride=1)
        self.upsample_model_15 = UpsampleBlock(96, 64, stride=1)

    def forward(self, x, x1, x2, x3):
        X00 = self.residual_model_head(x) + self.residual_model_head1(x1)  # ---   182 ~ 185
        # X10 = self.residual_model_head1(x1)

        X01 = self.residual_model_01(X00) + X00  # ---   208 ~ 211 ,AddBackward1213

        X10 = self.downsample_model_10(X00) + self.residual_model_head2(x2)  # ---   186 ~ 189
        X20 = self.downsample_model_20(X10) + self.residual_model_head3(x3)  # ---   190 ~ 193

        residual_11 = self.residual_model_11(X10) + X10  # 201 ~ 204    , sum  AddBackward1206
        downsample_11 = self.downsample_model_11(X01)  # 214 ~ 217
        X11 = residual_11 + downsample_11  # ---      AddBackward1218

        residual_21 = self.residual_model_21(X20) + X20  # 194 ~ 197  ,   sum  AddBackward1199
        downsample_21 = self.downsample_model_21(X11)  # 219 ~ 222
        X21 = residual_21 + downsample_21  # AddBackward1223

        X24 = self.residual_model_24(X21) + X21  # ---   224 ~ 227 , AddBackward1229
        X25 = self.residual_model_25(X24) + X24  # ---   230 ~ 233 , AddBackward1235

        upsample_14 = self.upsample_model_14(X24)  # 242 ~ 246
        residual_14 = self.residual_model_14(X11) + X11  # 248 ~ 251, AddBackward1253
        X14 = upsample_14 + residual_14  # ---   AddBackward1254

        upsample_04 = self.upsample_model_04(X14)  # 268 ~ 272
        residual_04 = self.residual_model_04(X01) + X01  # 274 ~ 277, AddBackward1279
        X04 = upsample_04 + residual_04  # ---  AddBackward1280

        upsample_15 = self.upsample_model_15(X25)  # 236 ~ 240
        residual_15 = self.residual_model_15(X14) + X14  # 255 ~ 258, AddBackward1260
        X15 = upsample_15 + residual_15  # AddBackward1261

        upsample_05 = self.upsample_model_05(X15)  # 262 ~ 266
        residual_05 = self.residual_model_05(X04) + X04  # 281 ~ 284,AddBackward1286
        X05 = upsample_05 + residual_05  # AddBackward1287

        X_tail = self.residual_model_tail(X05)  # 288 ~ 291

        return X_tail


class FeatureExtractor(nn.Module):
    """The quadratic model"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)
        self.prelu6 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x1 = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x1))
        x2 = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x2))
        x3 = self.prelu6(self.conv6(x))

        return x1, x2, x3


class AnimeInterp(nn.Module):
    """The quadratic model"""

    def __init__(self):
        super().__init__()
        self.feat_ext = FeatureExtractor()
        self.synnet = GridNet(6, 64, 128, 96 * 2, 3)

    def forward(self, I1, I2, F1t, F2t, Z1, Z2, feat1_pyramid, feat2_pyramid):
        feat11, feat12, feat13 = feat1_pyramid
        feat21, feat22, feat23 = feat2_pyramid

        I1t = warp(I1, F1t, Z1, strMode='soft')  #
        I2t = warp(I2, F2t, Z2, strMode='soft')  #
        one = torch.ones(I1.size(), requires_grad=True).to(device)
        norm1 = warp(one, F1t.clone(), Z1, strMode='soft')
        norm2 = warp(one, F2t.clone(), Z2, strMode='soft')
        norm_process(I1t, norm1)
        norm_process(I2t, norm2)
        # I1t[norm1 > 0] = I1t.clone()[norm1 > 0] / norm1[norm1 > 0]
        # I2t[norm2 > 0] = I2t.clone()[norm2 > 0] / norm2[norm2 > 0]

        F1td = dflow(F1t, feat11)
        F2td = dflow(F2t, feat21)
        Z1d = dmetric(Z1, feat11)
        Z2d = dmetric(Z2, feat21)
        feat1t1 = warp(feat11, F1td, Z1d, strMode='soft')
        feat2t1 = warp(feat21, F2td, Z2d, strMode='soft')
        one = torch.ones(feat11.size(), requires_grad=True).to(device)
        norm1 = warp(one, F1td.clone(), Z1d, strMode='soft')
        norm2 = warp(one, F2td.clone(), Z2d, strMode='soft')
        norm_process(feat1t1, norm1)
        norm_process(feat2t1, norm2)
        # feat1t1[norm1 > 0] = feat1t1.clone()[norm1 > 0] / norm1[norm1 > 0]
        # feat2t1[norm2 > 0] = feat2t1.clone()[norm2 > 0] / norm2[norm2 > 0]

        F1td = dflow(F1t, feat12)
        F2td = dflow(F2t, feat22)
        Z1d = dmetric(Z1, feat12)
        Z2d = dmetric(Z2, feat22)
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')
        one = torch.ones(feat12.size(), requires_grad=True).to(device)
        norm1 = warp(one, F1td.clone(), Z1d, strMode='soft')
        norm2 = warp(one, F2td.clone(), Z2d, strMode='soft')
        norm_process(feat1t2, norm1)
        norm_process(feat2t2, norm2)
        # feat1t2[norm1 > 0] = feat1t2.clone()[norm1 > 0] / norm1[norm1 > 0]
        # feat2t2[norm2 > 0] = feat2t2.clone()[norm2 > 0] / norm2[norm2 > 0]

        F1td = dflow(F1t, feat13)
        F2td = dflow(F2t, feat23)
        Z1d = dmetric(Z1, feat13)
        Z2d = dmetric(Z2, feat23)
        feat1t3 = warp(feat13, F1td, Z1d, strMode='soft')
        feat2t3 = warp(feat23, F2td, Z2d, strMode='soft')
        one = torch.ones(feat13.size(), requires_grad=True).to(device)
        norm1 = warp(one, F1td.clone(), Z1d, strMode='soft')
        norm2 = warp(one, F2td.clone(), Z2d, strMode='soft')
        norm_process(feat1t3, norm1)
        norm_process(feat2t3, norm2)
        # feat1t3[norm1 > 0] = feat1t3.clone()[norm1 > 0] / norm1[norm1 > 0]
        # feat2t3[norm2 > 0] = feat2t3.clone()[norm2 > 0] / norm2[norm2 > 0]

        # with torch.autocast("cuda"):
        with torch.cuda.amp.autocast(enabled=True):
            It_warp = self.synnet(torch.cat([I1t, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
                                  torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return It_warp.float()


