#include <cstdio>

__global__ void hello() {
    printf("thread %d reporting\n", threadIdx.x);
}

int main() {
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    printf("build system works\n");
    return 0;
}