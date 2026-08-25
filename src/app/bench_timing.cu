#include <cstdio>
#include "gpu_timer.hpp"
#include "timer.hpp"
__global__ void busy(float *out, int n, int iters){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = 0.0f;
    for ( int k = 0; k < iters; k++){
        x = x * 1.001f + 0.5f;
    }
    out[i] = x;
}

int main(){
    int n = 1 << 20; //1,048,567 elements
    float *out;
    cudaMallocManaged(&out, n * sizeof(float));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    GpuTimer t;
    t.start(0);
    busy<<<blocks, threads>>>(out, n, 4000);
    t.stop(0);
    printf("gpu: %.3f ms\n", t.elapsed());
    cudaFree(out);



        // CPU timer, no sync — measures only the launch call
    Timer cpu;
    cpu.start();
    busy<<<blocks, threads>>>(out, n, 4000);
    double no_sync = cpu.stop();
    cudaDeviceSynchronize();          // let it finish before the next test

    // CPU timer, with sync — measures the work, plus scheduler overhead
    cpu.start();
    busy<<<blocks, threads>>>(out, n, 4000);
    cudaDeviceSynchronize();
    double with_sync = cpu.stop();

    printf("cpu no sync   : %.3f ms\n", no_sync);
    printf("cpu with sync : %.3f ms\n", with_sync);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launch error: %s\n", cudaGetErrorString(err));
    }
    return 0;


}