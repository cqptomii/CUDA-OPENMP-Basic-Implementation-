#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

__global__ void vecAdd(float* d_vec1, float* d_vec2, float* d_result) {
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    d_result[index] = d_vec1[index] + d_vec2[index];
}
void cudaDeviceInit() {
    int devCount, device;
    cudaGetDeviceCount(&devCount);
    if (devCount < 1) {
        printf("No CUDA capable devices dectected. \n");
        exit(EXIT_FAILURE);
    }
    for (device = 0; device < devCount; ++device) {
        cudaDeviceProp dev_props;
        cudaGetDeviceProperties(&dev_props, device);
        printf(" Device %d has compute capability %d.%d.\n", device, dev_props.major, dev_props.minor);
        if (dev_props.major > 1 || (dev_props.major == 1 && dev_props.minor >= 3))
            break;
        if (device == devCount) {
            printf("No device above 1.2 capability detected. \n");
            exit(EXIT_FAILURE);
        }
        else {
            cudaSetDevice(device);
        }
    }
}

int main(int argc, char** argv) {

    cudaDeviceInit();
    int i, N = 720896;
    srand(time(nullptr));

    cudaEvent_t start, stop;
    float time_elapsed = 0;
    // Host vector
    float* h_vec1;
    float* h_vec2;
    float* h_result;

    // initialize Host variables
    h_vec1 = (float*)malloc(sizeof(float) * N);
    h_vec2 = (float*)malloc(sizeof(float) * N);
    h_result = (float*)malloc(sizeof(float) * N);

    // Device copies of vec
    float* d_vec1;
    float* d_vec2;
    float* d_result;


    // initialize Device variables
    cudaMalloc((void**)&d_vec1, sizeof(float) * N);
    cudaMalloc((void**)&d_vec2, sizeof(float) * N);
    cudaMalloc((void**)&d_result, sizeof(float) * N);

    if (argc > 1)
        N = atoi(argv[1]);
    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running GPU vecAdd for %i elements \n", N);
    
    cudaEventRecord(start, 0);

    /* generate random data */
    for (i = 0; i < N; i++)
    {
        h_vec1[i] = (float)rand() / RAND_MAX;
        h_vec2[i] = (float)RAND_MAX - h_vec1[i];
    }
    
    // Copy variables from host to device memory
    cudaMemcpy(d_vec1, h_vec1,sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, sizeof(float) * N, cudaMemcpyHostToDevice);

    vecAdd <<<N / 512, 512>>> (d_vec1,d_vec2,d_result);

    cudaMemcpy(d_result, h_result, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    time_elapsed /= 1000;

    printf("time=%.4f seconds, MFLOPS=%.1f\n", time_elapsed, (float)N / time_elapsed / 1E6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* print out first 10 results */
    for (i = 0; i < 10; i++)
        printf("h_result[%i]=%.2f\n", i, h_result[i]);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);
    free(h_vec1);
    free(h_vec2);
    free(h_result);

    exit(EXIT_SUCCESS);
}