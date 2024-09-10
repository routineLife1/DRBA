# Distance Ratio Based Adjuster for Animeinterp

> **Abstract：** This project serves as a control mechanism for Video Frame Interpolation (VFI) networks specifically
> tailored for anime.
> By calculating the DistanceRatioMap, it adjusts the frame interpolation strategies for spatiotemporally nonlinear and
> linear regions,
> thereby preserving the original pace and integrity of the characters while avoiding distortions common in frame
> interpolation.

# Demo

## input

![input](https://github.com/hyw-dev/FCLAFI/assets/68835291/cc9fb083-0f8d-48e1-b33e-0a893f313329)

## output

![output](https://github.com/hyw-dev/FCLAFI/assets/68835291/5138f267-6904-42ce-9551-b0891812a650)

# Version Comparison

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

# Demos Videos

**Coming soon.**