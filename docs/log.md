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

## Aug 23
- Wrote CPU timer
- Tested CPU timer

## Aug 25
- Wrote GPU Timer class
- Tested GPU timer
- BUG - freed the buffer before use, launched failed silently

## Aug 26
- Instrumented capture loop working. 300 frames, 640x480.
- p50=32.05ms p95=36.07ms p99=36.30ms
- p50 ≈ camera frame period (~31fps), not processing cost. Loop does
  nothing but grab, so it blocks waiting for the sensor. This number
  can't be optimized away.
- 4ms p50→p99 spread = mild jitter, no dropped frames.
- channels=3 confirms OpenCV converts YUYV→BGR on CPU every frame.
  Baseline cost the fused kernel should eliminate.
- Expect this number to DROP when inference is added — frames will
  already be queued. Not an improvement, just a regime change.
- Exported yolov8n to ONNX, opset 12, 640x640. Output shape (1, 84, 8400):
  84 = 4 box coords + 80 COCO class scores. Layout is [84][8400], transposed
  from the obvious ordering — matters for postprocess indexing.
- Built FP16 TensorRT engine with trtexec.
- Baseline: GPU compute 4.39ms mean, throughput 227 qps.
- H2D 0.288ms + D2H 0.226ms per frame — trtexec copies because it's generic.
  Zero-copy design should eliminate ~0.5ms/frame.
- trtexec warned GPU compute unstable, CoV 3.4%. Locked clocks with
  jetson_clocks, mode MAXN_SUPER (2). Doesn't persist across reboot.

## Aug 28
- g++ doesn't get CUDA include paths automatically like nvcc does.
  Needed find_package(CUDAToolkit) + ${CUDAToolkit_INCLUDE_DIRS} for
  any .cpp that includes TensorRT headers.