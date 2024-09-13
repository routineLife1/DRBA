# Distance Ratio Based Adjuster for Animeinterp

> **Abstract：** This project serves as a control mechanism for Video Frame Interpolation (VFI) networks specifically
> tailored for anime.
> By calculating the DistanceRatioMap, it adjusts the frame interpolation strategies for spatiotemporally nonlinear and
> linear regions,
> thereby preserving the original pace and integrity of the characters while avoiding distortions common in frame
> interpolation.

# 👀Demos Videos

**Coming soon.**

## 🔧Dependencies
**Set up the environment for the following repository**
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [FastFlowNet](https://github.com/ltkong218/FastFlowNet)
- [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)

## ⚡Usage 

**Coming soon.**

# 📖Version Comparison

## gmfss(For learning and reference only.)

## rife_lite(Aimed at real-time playback.)

**Due to the limitations of the RIFE algorithm's performance, some distortion may occur in the background during
compensation.**

![rife](https://github.com/user-attachments/assets/e0480165-c748-43ac-ad3c-5e6fb7adea7f)

**If you use FastFlowNet TRT, RIFE v4.22 Lite TRT, and implement the project with VapourSynth, real-time playback can be
achieved on a reasonably powerful NVIDIA GPU, such as the RTX 3070.**

## gmfss_union(For high-quality output.)

**Combining the strengths of both RIFE and GMFSS, it delivers outstanding results.**

![gmfss](https://github.com/user-attachments/assets/5a4ca540-ddfa-4a93-ab21-e39eb9299e89)
