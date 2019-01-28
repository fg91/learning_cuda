#include <stdio.h>
#include <iostream>
#include "gputimer.h"

using std::cout;
using std::endl;

const int N = 1024;  // matrix size is NxN
const int K = 32;    // tile size is KxK

void fill_matrix(float *mat) {
  for (int i = 0; i < N * N; i++) {
    mat[i] = (float) i;
  }
}

void print_matrix(float *mat) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      printf("%4.4g ", mat[i + j * N]);  // row major mode with rows j and cols i
    }
    printf("\n");
  }
}

void transpose_CPU(float in[], float out[]) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      out[j + i * N] = in[i + j * N];
    }
  }
}

int compare_matrices(float *gpu, float *ref) {
  int result = 0;
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      if (ref[i + j * N] != gpu[i + j * N]) result = 1;
    }
  }
  return result;
}

// -------------------------------------------- //
// ----Different CUDA kernels for transpose---- //
// -------------------------------------------- //

// Kernel 1:
__global__ void transpose_serial(float in[], float out[]) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      out[j + i * N] = in[i + j * N];
    }
  }
}

// Kernel 2:
__global__ void transpose_parallel_per_row(float in[], float out[]) {
  int i = threadIdx.x;

  // one thread per row
  for (int j = 0; j < N; j++) {
      out[j + i * N] = in[i + j * N];
  }
}

// Kernel 3:
__global__ void transpose_parallel_per_element(float in[], float out[]) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // one thread per element
  out[j + i * N] = in[i + j * N];

}



// -------------------------------------------- //
// -------------------------------------------- //
// -------------------------------------------- //


int main() {

  int numbytes = N * N * sizeof(float);

  float *in = (float *) malloc(numbytes);
  float *out = (float *) malloc(numbytes);
  float *gold = (float *) malloc(numbytes);

  fill_matrix(in);
  transpose_CPU(in, gold);

  // device pointers
  float *d_in, *d_out;

  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);
  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

  GpuTimer timer;

  /*
   * Now time each kernel and verify that it produces the correct result.
   *
   * To be really careful about benchmarking purposes, we should run every kernel once
   * to "warm" the system and avoid any compilation or code-caching effects, then run
   * every kernel 10 or 100 times and average the timings to smooth out any variance.
   * But this makes for messy code and our goal is teaching, not detailed benchmarking.
   */

  /*
  // Kernel 1:
  timer.Start();
  transpose_serial<<<1, 1>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_serial: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");
  */
  // Kernel 2:
  timer.Start();
  transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_parallel_per_row: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

  // Kernel 3:
  dim3 blocks(N/K, N/K);
  dim3 threads(K, K);
  
  timer.Start();
  transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_parallel_per_element: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

  
  return 0;
}
