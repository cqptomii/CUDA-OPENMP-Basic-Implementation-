
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <assert.h>

#define SOFTENING 1e-9f


inline cudaError_t checkCuda(cudaError_t result){
    if(result != cudaSuccess){
        fprintf(stderr,"%s \n",cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
__global__
void bodyForce(Body *p, float dt, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n;i += stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

__global__
void integrateBodies(Body *p,float dt, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx ; i < n; i += stride) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}


int main(const int argc, const char** argv) {
    int deviceId;
    int num_sm;
    int warpSize;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_sm,cudaDevAttrMultiProcessorCount,deviceId);
    cudaDeviceGetAttribute(&warpSize,cudaDevAttrWarpSize,deviceId);

  int nBodies = 2<<15;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  cudaMallocManaged(&buf,bytes);
  
  Body *p = (Body*)buf;

  cudaMemPrefetchAsync(buf,bytes,cudaCpuDeviceId);
 
  read_values_from_file(initialized_values, buf, bytes);

  double totalTime = 0.0;


  cudaMemPrefetchAsync(buf,bytes,deviceId);


  int threads_per_block = 256;
  int num_blocks = num_sm * warpSize;

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();


    bodyForce<<<num_blocks,threads_per_block>>>(p, dt, nBodies); // compute interbody forces

    integrateBodies<<<num_blocks,threads_per_block>>>(p,dt,nBodies);

    cudaDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }
  cudaMemPrefetchAsync(p,bytes,cudaCpuDeviceId);

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(buf);
}