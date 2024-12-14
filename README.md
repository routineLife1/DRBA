# Distance Ratio Based Adjuster for Animeinterp

> **Abstract：** This project serves as a control mechanism for Video Frame Interpolation (VFI) networks specifically
> tailored for anime.
> By calculating the DistanceRatioMap, it adjusts the frame interpolation strategies for spatiotemporally nonlinear and
> linear regions,
> thereby preserving the original pace and integrity of the characters while avoiding distortions common in frame
> interpolation.

# 👀Demo

## input
![input](https://github.com/hyw-dev/FCLAFI/assets/68835291/cc9fb083-0f8d-48e1-b33e-0a893f313329)
## output
![output](https://github.com/hyw-dev/FCLAFI/assets/68835291/5138f267-6904-42ce-9551-b0891812a650)

# 👀Demos Videos

**[OP「つよがるガール」](https://www.bilibili.com/video/BV1uJtPe9EdY/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP1「勇者」](https://www.bilibili.com/video/BV12QsaeREmr/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP2「放晴」](https://www.bilibili.com/video/BV1RYs8eFE77/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

## 🔧Installation

```bash
git clone https://github.com/routineLife1/DRBA.git
cd DRBA
pip3 install -r requirements.txt
```
The cupy package is included in the requirements, but its installation is optional. It is used to accelerate computation. If you encounter difficulties while installing this package, you can skip it.

## ⚡Usage 

**Video Interpolation**
```bash
  # For speed preference
  python interpolate_video_rife_anyfps.py -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3 -hw
  # For quality preference
  python interpolate_video_gmfss_union_anyfps.py -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3 -hw
```

**Full Usage**
```bash
Usage: python interpolate_video_rife_anyfps.py -i in_video -o out_video [options]...
       
  -h                   show this help
  -i input             input video path (absolute path of output video)
  -o output            output video path (absolute path of output video)
  -fps dst_fps         target frame rate (default=60)
  -s enable_scdet      enable scene change detection (default Enable)
  -st scdet_threshold  ssim scene detection threshold (default=0.3)
  -hw hwaccel          enable hardware acceleration encode (default Enable) (require nvidia graph card)
  -s scale             scale factor (default=1.0) flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution
```

- input accept absolute video file path. Example: E:/input.mp4
- output accept absolute video file path. Example: E:/output.mp4
- dst_fps = target interpolated video frame rate. Example: 60
- enable_scdet = enable scene change detection.
- scdet_threshold = scene change detection threshold. The smaller the value, the more sensitive the detection.
- hwaccel = enable hardware acceleration during encoding output video.
- scale = flow scale factor. Decrease this value to reduce the computational difficulty of the model at higher resolutions. Generally, use 1.0 for 1080P and 0.5 for 4K resolution.

# 📖Version Comparison

## gmfss(For learning and reference only.)
The explanation of the algorithm's principles has not been organized yet. You may reach out in issue if you have any questions regarding the details for now.

## rife(Aimed at real-time playback.)

**Due to the limitations of the RIFE algorithm's performance, some distortion may occur in the background during
compensation.**

![rife](https://github.com/user-attachments/assets/e0480165-c748-43ac-ad3c-5e6fb7adea7f)

**If you use RIFE v4.26 TRT, and implement the project with VapourSynth, real-time playback can be achieved on a reasonably powerful NVIDIA GPU.**

## gmfss_union(For high-quality output.)

**Combining the strengths of both RIFE and GMFSS, it delivers outstanding results.**

![gmfss](https://github.com/user-attachments/assets/5a4ca540-ddfa-4a93-ab21-e39eb9299e89)
