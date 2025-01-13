# Distance Ratio Based Adjuster for Animeinterp

> **介绍：** 本项目可被视作一种专为动漫视频帧插值网络（Anime VFI）定制的控制机制。通过计算距离比率图（DistanceRatioMap）来调整视频中非线性与线性运动区域的帧插值策略，以达到流畅补帧的同时，保持动漫视频的原始节奏和人物绘图的完整性。

<a href="https://colab.research.google.com/drive/1BGlSg7ghPoXC_s5UuF8Z__0YV4fGrQoA?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
# 👀效果演示
https://github.com/user-attachments/assets/ec1dc508-8580-4259-9e9b-c25399d69579

## 🔧部署项目

```bash
git clone https://github.com/routineLife1/DRBA.git
cd DRBA
pip3 install -r requirements.txt
```
requirement中包含cupy包依赖, 该包用于加速计算。如果安装遇到问题，可以跳过，这通常不会对运行项目造成影响。
## ⚡用法 

**视频补帧**
```bash
  # 速度优先(致力于实时播放)
  python infer.py -m rife -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3 -hw
  # 质量优先
  python infer.py -m gmfss_union -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3 -hw
```

**完整用法**
```bash
Usage: python infer.py -m model -i in_video -o out_video [options]...
       
  -h                   展示此帮助信息
  -m model             选择使用的模型 (目前支持: rife, gmfss, gmfss_union) (默认为rife)
  -i input             输入视频的绝对路径(例: E:/input.mp4)
  -o output            输出视频的绝对路径(例: E:/output.mp4)
  -fps dst_fps         导出视频的目标帧率 (默认为60)
  -t times             导出视频的帧率倍率 (默认为-1, 若指定则优先使用倍率模式)
  -s enable_scdet      开启转场识别 (默认关闭)
  -st scdet_threshold  ssim转场识别阈值 (默认为0.3)
  -hw hwaccel          开启硬件加速编码 (默认关闭) (需要NVIDIA显卡)
  -s scale             光流缩放尺度 (默认为1.0), 通常在处理1080p分辨率视频使用1.0, 4K分辨率时使用0.5
```

- scdet_threshold: 转场识别阈值. 该数值越大, 识别越敏感
- scale: 光流缩放尺度. 缩小该值可以降低网络在处理大分辨率时的复杂度.

# 👀其他效果演示

**[OP「つよがるガール」](https://www.bilibili.com/video/BV1uJtPe9EdY/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP1「勇者」](https://www.bilibili.com/video/BV12QsaeREmr/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP2「放晴」](https://www.bilibili.com/video/BV1RYs8eFE77/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

## 📖技术概览
DRBA由两部分组成('计算DRM图' 和 '将DRM图应用到补帧网络'), 输入三帧, 输出调整后的帧插值.
![Overview](assert/Overview.png)

# 🔗参考
光流算法: [GMFlow](https://github.com/haofeixu/gmflow)

补帧算法: [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) [MultiPassDedup](https://github.com/routineLife1/MultiPassDedup)
