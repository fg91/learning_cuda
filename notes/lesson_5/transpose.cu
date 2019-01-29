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

  /*
    Reading: threads with adjacent thread idx in x are reading adjacent values of the input matrix.
             good
    Writing: threads with adjacent thread idx in x are writing to strided memory locations
             with stride factor N. Very bad!!
   */

}

// Kernel 4:
__global__ void transpose_parallel_per_element_tiled(float in[], float out[]) {
  // (i,j) locations of the tile corners for input and output matrices
  int in_corner_i = blockIdx.x * K;
  int in_corner_j = blockIdx.y * K;

  // out corners simple reverse y and x
  int out_corner_i = blockIdx.y * K;
  int out_corner_j = blockIdx.x * K;

  // which element of the tile to read from and write to
  int x = threadIdx.x;
  int y = threadIdx.y;

  __shared__ float tile[K][K];

  // coalesced read from global memory to write into shared memory
  tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];

  __syncthreads();

  // read from shared memory, coalesced write into global memory
  out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];

  // adjacent threads in x write to adjacent memory locations. Good!!
}

// Kernel 5:
__global__ void transpose_parallel_per_element_tiled16(float in[], float out[]) {
  // (i,j) locations of the tile corners for input and output matrices
  int in_corner_i = blockIdx.x * 16;
  int in_corner_j = blockIdx.y * 16;

  // out corners simple reverse y and x
  int out_corner_i = blockIdx.y * 16;
  int out_corner_j = blockIdx.x * 16;

  // which element of the tile to read from and write to
  int x = threadIdx.x;
  int y = threadIdx.y;

  __shared__ float tile[16][16];

  // coalesced read from global memory to write into shared memory
  tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];

  __syncthreads();

  // read from shared memory, coalesced write into global memory
  out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];

  // adjacent threads in x write to adjacent memory locations. Good!!
}

// Kernel 6:
__global__ void transpose_parallel_per_element_tiled16_padded(float in[], float out[]) {
  // (i,j) locations of the tile corners for input and output matrices
  int in_corner_i = blockIdx.x * 16;
  int in_corner_j = blockIdx.y * 16;

  // out corners simple reverse y and x
  int out_corner_i = blockIdx.y * 16;
  int out_corner_j = blockIdx.x * 16;

  // which element of the tile to read from and write to
  int x = threadIdx.x;
  int y = threadIdx.y;

  // shared memory padding (reserving more than necessary) to reduce shared memory bank conflicts
  __shared__ float tile[16][16+1];

  // Coalesced read from global memory to write into shared memory
  tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];

  __syncthreads();

  // read from shared memory, coalesced write into global memory
  out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];

  // adjacent threads in x write to adjacent memory locations. Good!!
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

  // Kernel 1:
  timer.Start();
  transpose_serial<<<1, 1>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_serial: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

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

  // Kernel 4:
  timer.Start();
  transpose_parallel_per_element_tiled<<<blocks, threads>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_parallel_per_element_tiled: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

  // Kernel 5:
  dim3 blocks16(N/16, N/16);
  dim3 threads16(16, 16);

  timer.Start();
  transpose_parallel_per_element_tiled16<<<blocks16, threads16>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_parallel_per_element_tiled16: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");

  // Kernel 6:
  timer.Start();
  transpose_parallel_per_element_tiled16_padded<<<blocks16, threads16>>>(d_in, d_out);
  timer.Stop();
  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
  printf("Transpose_parallel_per_element_tiled16_padded: %g ms, %s\n",
         timer.Elapsed(), compare_matrices(out, gold) ? "Failed" : "Success");
  
  
  return 0;
}
