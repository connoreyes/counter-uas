#pragma once
#include <cuda_runtime.h>

class GpuTimer{
public:
    GpuTimer(){
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer(){
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream){
        cudaEventRecord(start_, stream);
    }
    void stop(cudaStream_t stream){
        cudaEventRecord(stop_, stream);
    }
    float elapsed(){
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
        
    }

private:
    cudaEvent_t start_, stop_;
};