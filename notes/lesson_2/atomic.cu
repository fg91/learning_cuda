// "Copyright 2018 <Fabio M. Graetz>"
#include <stdio.h>
#include <iostream>
#include "./gputimer.h"

using std::cout;
using std::endl;

#define NUM_THREADS 1000000
#define ARRAY_SIZE 10
#define BLOCK_WIDTH 1000

void print_array(int *arr, int size) {
  for (int i = 0; i < size; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

__global__ void increment_naive(int *g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  i = i % ARRAY_SIZE;
  g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = i % ARRAY_SIZE;
  atomicAdd(& g[i], 1);  // pointer in device memory
}

int main() {
  GpuTimer timer;
  cout << NUM_THREADS << " total threads in " << NUM_THREADS / BLOCK_WIDTH
       << " blocks writing into " << ARRAY_SIZE << " elements" << endl;

  // declare and allocate host memory
  int h_array[ARRAY_SIZE];
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

  // declare, allocate, and zero out device memory
  int *d_array;
  cudaMalloc(&d_array, ARRAY_BYTES);
  cudaMemset(d_array, 0, ARRAY_SIZE);

  // launch the kernel
  timer.Start();
  increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
  // increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
  timer.Stop();

  // copy array to host
  cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  print_array(h_array, ARRAY_SIZE);
  cout << "Time elapsed = " << timer.Elapsed() << " ms" << endl;

  // free device memory allocation
  cudaFree(d_array);
  return 0;
}
