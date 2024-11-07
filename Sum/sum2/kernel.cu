
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sum(double* d_a, double* d_s)
{
    __shared__ double temp[512];
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    temp[threadIdx.x] = d_a[global_index];

    __syncthreads();

    if (threadIdx.x == 0) {
        double summ = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            summ += temp[i];
        }
        atomicAdd(d_s, summ);
    }
}
void cuda_init_device() {
    int dev_count, device = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        printf("No cuda device detected \n");
        exit(EXIT_FAILURE);
    }
    for (int device = 0; device < dev_count; ++device) {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, device);
        if (dev_prop.major < 1 || (dev_prop.major == 1 && dev_prop.minor >= 3)) {
            break;
        }

        if (device == dev_count) {
            printf("No device with 1.3 compute capability was found \n");
            exit(EXIT_FAILURE);
        }
        else {
            cudaSetDevice(device);
        }
    }
}
int  main(int argc, char** argv)
{
    long long int i, N = 2097152;

    cudaEvent_t start, stop;

    float time_elapsed = 0.0f;
    float gflop = 0.0f;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    cuda_init_device();

    // Initialize host variables
    double* h_a;
    double h_s = 0;
    h_a = (double*)malloc(N * sizeof(double));

    srand(0);

    for (i = 0; i < N; i++) {  // generate random data
        h_a[i] = (double)rand() / RAND_MAX;
    }

    printf("Running CPU sum for %lld elements\n", N);

    // Initialize device variables
    double* d_a;
    double* d_s;
    cudaMalloc((void**)&d_a, sizeof(double) * N);
    cudaMalloc((void**)&d_s, sizeof(double));
    cudaMemcpy(d_a, h_a, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Specify amount of block and threads

    int threads = 512;
    int block_amount = (N + threads - 1) / threads;
    cudaEventRecord(start, 0);

    sum <<<block_amount, threads >> > (d_a, d_s);  // call compute kernel

    // Copu sum value from device to host memory

    cudaMemcpy(&h_s, d_s, sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("sum=%.2f\n", h_s);

    gflop = ((N - 1) / time_elapsed / 1E3f);
    printf("sec = %f   GFLOPS = %.3f\n", time_elapsed / 1E3F, gflop);

    cudaFree(d_a);
    cudaFree(d_s);
    free(h_a);  // free allocated memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    scanf("%d");
    return EXIT_SUCCESS;
}

