
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16
// a = b * c
__global__ void mmult(float* a, float* b, float* result, int N){

    // Grid index
    int indexX = blockDim.x * blockIdx.x + threadIdx.x;
    int indexY = blockDim.y * blockIdx.y + threadIdx.y;
    float temp_summ = 0.0f;
    __shared__ float temp_matrix_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_matrix_b[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = 0; i < N / BLOCK_SIZE; ++i) {
        temp_matrix_a[threadIdx.y][threadIdx.x] = a[indexY + i * BLOCK_SIZE + threadIdx.x];
        temp_matrix_b[threadIdx.y][threadIdx.x] = b[indexX + (i * BLOCK_SIZE + threadIdx.y) * N];
        
        __syncthreads();
        
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            temp_summ += temp_matrix_a[threadIdx.y][j] * temp_matrix_b[j][threadIdx.x];
        }

        // second synchronisation to ensure that all threads in one block compute the summ
        __syncthreads();
    }
    
    result[indexY*N + indexX] = temp_summ;
}
// a[][] = 0, b[][] = 1; c[][] = 1
void minit(float* a, float* b, float* result, int N)
{
    int i, j;

    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
        {
            a[i + N * j] = 1.0f;
            b[i + N * j] = 1.0f;
            result[i + N * j] = 0.0f;
        }
}

void mprint(float* a, int N, int M)
{
    int i, j;

    for (j = 0; j < M; j++)
    {
        for (i = 0; i < M; i++)
            printf("%.2f ", a[i + N * j]);
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc,char **argv)
{
   // Width of the matrix
    long int N = 2048;
    float time_elapsed = 0;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    //Cuda timer
    cudaEvent_t start, stop;
 
    //Instantiate host matrix
    float *h_matrix_a = (float*)malloc(sizeof(float) * N * N);
    float *h_matrix_b = (float*)malloc(sizeof(float) * N * N);
    float *h_matrix_result = (float*)malloc(sizeof(float) * N * N);

    minit(h_matrix_a, h_matrix_b, h_matrix_result, N);
    //Instantiate device matrix
    float* d_matrix_a, * d_matrix_b, * d_matrix_result;

    cudaMalloc((void**)&d_matrix_a, sizeof(float) * N * N);
    cudaMalloc((void**)&d_matrix_b, sizeof(float) * N * N);
    cudaMalloc((void**)&d_matrix_result, sizeof(float) * N * N);

    //Initalisation

    cudaMemcpyAsync(d_matrix_a, h_matrix_a, sizeof(float) * N * N, cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_matrix_b, h_matrix_b, sizeof(float) * N * N, cudaMemcpyHostToDevice,0);
    cudaMemsetAsync(d_matrix_result, 0, sizeof(float) * N * N,0);

    //Grid dim and block dim

    dim3 block(BLOCK_SIZE,BLOCK_SIZE); // in this case each block process four cell of the result matrix
    dim3 grid(N / block.x, N / block.y);

    printf("Matrix size : %i * %i \n", N, N);
    printf("Grid size : %i %i \n", grid.x, grid.y);
    printf("Block size : %i %i %i \n", block.x, block.y, block.z);

    //Create cuda event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf(" Start GPU work \n");

    cudaEventRecord(start,0);

    // launch device function

    mmult<<<grid, block>>> (d_matrix_a,d_matrix_b,d_matrix_result,N);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_matrix_result, d_matrix_result, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    //print the matrix
    mprint(h_matrix_result, N, 5);

    cudaEventElapsedTime(&time_elapsed, start, stop);
    
    time_elapsed /= 1E3;

    printf("time = %.2f   GFLOPS = %.3f \n", time_elapsed, (float) N*N*N*2.0f / time_elapsed / 1E9);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_result);
    free(h_matrix_a);
    free(h_matrix_b);
    free(h_matrix_result);

    return EXIT_SUCCESS;
}
