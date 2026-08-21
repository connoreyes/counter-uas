## Aug 17
- Verified stack: CUDA 12.6, TensorRT 10.3, JetPack 6.2.1
- Confirmed integrated:1 on device query
- Camera: YUYV 640x480 30fps. Measured 23fps until auto-exposure
  disabled, then 28.6fps.

## Aug 19
- Wrote 4 CUDA kernels: vector add, multi-block add, 2D fill, resize
- CMake project builds, repo pushed to GitHub

## Aug 20
- Wrote Stat class: 512-slot ring buffer, percentiles
- Tested wrap behavior, p50 of 600 samples = 345 as predicted
