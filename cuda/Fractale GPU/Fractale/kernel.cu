
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#define NB_ITERATION 255
#define WIDTH 1024
#define HEIGHT 768
unsigned char iter(double c_r, double c_i);
void save_image(unsigned char* image, int width, int height, char* filename);
void make_fractal_cpu(unsigned char* image, int width, int heigth, double x_upper, double x_lower, double y_upper, double y_lower);
__device__ unsigned char iter_gpu(double c_r, double c_i);

__global__ void make_fractal_gpu1(unsigned char *image,double x_upper,double x_lower, double y_upper, double y_lower){
    unsigned int indexX = blockDim.x * blockIdx.x;
    unsigned int indexY = blockDim.y * blockIdx.y;

    double x_inc = (x_upper - x_lower) / WIDTH;
    double y_inc = (y_upper - y_lower) / HEIGHT;

    image[indexY * WIDTH + indexX] = iter_gpu(x_lower + indexX * x_inc, y_lower + indexY * y_inc);
}

__global__ void make_fractal_gpu2(unsigned char* image, double x_upper, double x_lower, double y_upper, double y_lower) {
    unsigned int indexX = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int indexY = blockDim.y * blockIdx.y + threadIdx.y;

    if (indexX < WIDTH && indexY < HEIGHT) {
        double x_inc = (x_upper - x_lower) / WIDTH;
        double y_inc = (y_upper - y_lower) / HEIGHT;

        image[indexY * WIDTH + indexX] = iter_gpu(x_lower + (indexX) * x_inc, y_lower + (indexY) * y_inc);
    }
}
// Fonction only on GPU
__device__ unsigned char iter_gpu(double c_r, double c_i) {
    unsigned char i = 0;
    double z_i_temp, z_r_temp;
    double z_i = 0.0, z_r = 0.0;
    
    while (((z_i * z_i + z_r * z_r) < 4.0) && (i++ < NB_ITERATION)) {
        z_r_temp = z_r * z_r - z_i * z_i;
        z_i_temp = 2 * z_r * z_i;

        z_r = c_r + z_r_temp;
        z_i = c_i + z_i_temp;
    }

    return i;
}

int main(int argc, char **argv){
    double x_upper = 1.000000000000, x_lower = -2.100000000000, y_upper = 1.300000000000, y_lower = -1.300000000000; 
    float time_elapsed = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned char* h_image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char*));

    cudaEventRecord(start, 0);

    make_fractal_cpu(h_image, WIDTH, HEIGHT, x_upper, x_lower, y_upper, y_lower);
    
    // Time handling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("Generate fractal in %.2f seconds with CPU\n\n",time_elapsed / 1E3);

    save_image(h_image, WIDTH, HEIGHT, "fractal_cpu.ppm");


    //
    //  GPU
    //

    unsigned char* d_image;
    cudaMalloc((void**)&d_image, sizeof(unsigned char*) * WIDTH * HEIGHT);
        
    dim3 block(1);
    dim3 grid(WIDTH, HEIGHT);

    cudaEventRecord(start, 0);

    printf("Fractal with blockdim of 1 Thread \n\n");
    make_fractal_gpu1<<<grid, block >> > (d_image, x_upper, x_lower, y_upper, y_lower);

    // Time handling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("block size : %i %i %i \n", block.x, block.y, block.z);
    printf("grid size : %i %i \n", grid.x, grid.y);
    printf("Generated fractal in %.4f seconds with GPU \n\n\n",time_elapsed / 1E3F);

    cudaMemcpy(h_image, d_image, sizeof(unsigned char*) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    save_image(h_image, WIDTH, HEIGHT, "fractale_gpu1.ppm");


    /// with block of 16x16 threads
    dim3 blocksize(16, 16);
    dim3 gridsize((WIDTH + blocksize.x-1) / blocksize.x, (HEIGHT + blocksize.y-1) / blocksize.y);

    cudaEventRecord(start, 0);
    make_fractal_gpu2<<<gridsize,blocksize>>>(d_image, x_upper, x_lower, y_upper, y_lower);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("Fractal with blockdim of %i x %i THREADS \n\n", blocksize.x, blocksize.y);
    printf("block size : %i %i %i \n", block.x, block.y, block.z);
    printf("grid size : %i %i \n", grid.x, grid.y);
    printf("Generated fractal in %.4f seconds with GPU \n\n\n", time_elapsed / 1E3F);
    
    cudaMemcpy(h_image, d_image, sizeof(unsigned char*) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    
    save_image(h_image, WIDTH, HEIGHT, "fractale_gpu16.ppm");


    

    // Memory free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_image);
    cudaFree(d_image);

    scanf("");
    
    return EXIT_SUCCESS;
}



void save_image(unsigned char* image, int width, int height, char* filename){
    FILE* fd = NULL;
    int i, x, y;
    struct { int r, g, b; } colors[256];

    for (i = 0; i < 256; i++)
    {
        colors[i].r = abs(((i + 60) % 256) - 127);
        colors[i].g = abs(((i + 160) % 256) - 127);
        colors[i].b = abs(((i + 200) % 256) - 127);
    }


    fd = fopen(filename, "w");

    fprintf(fd, "P3\n%d %d\n255\n", width, height);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            int pixel = image[y * width + x];
            fprintf(fd, "%i %i %i\n", colors[pixel].r, colors[pixel].g, colors[pixel].b);
        }
    }

    fclose(fd);
}
unsigned char iter(double c_r, double c_i) {

    unsigned char i = 0;
    double z_i_temp, z_r_temp;
    double z_i = 0, z_r = 0;

    while (((z_i*z_i + z_r*z_r) < 4.0) && (i++ < NB_ITERATION)) {
        z_r_temp = z_r * z_r - z_i * z_i;
        z_i_temp = 2 * z_r * z_i;

        z_r = c_r + z_r_temp;
        z_i = c_i + z_i_temp;
    }

    return i;
}
void make_fractal_cpu(unsigned char* image, int width, int heigth, double x_upper, double x_lower, double y_upper, double y_lower) {

    // deux index pour diviser la distance entre le module et la taille de l'image
    double x_inc = (x_upper - x_lower) / width;
    double y_inc = (y_upper - y_lower) / heigth;

    for (int i = 0; i < heigth; ++i) {
        for (int j = 0; j < width; ++j) {
            image[i * width + j] = iter((x_lower + j*x_inc),(y_lower + i*y_inc));
        }
    }
}
