// "Copyright 2019 <Fabio M. Graetz>"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

template<class T>
__device__ T scanHillisSteele(T *keys) {
  unsigned int i = threadIdx.x;
  unsigned int n = blockDim.x;

  for (int offset = 1; offset < n; offset <<=1) {
    T temp;
    if (i >= offset) {
      temp = keys[i - offset];
    }

    __syncthreads();

    if (i >= offset) {
      keys[i] = temp + keys[i];
    }

    __syncthreads();
  }

  return keys[i];
}

__global__ void partition_by_bit(unsigned int *keys, double *values, unsigned int bit) {
  unsigned int i = threadIdx.x;
  unsigned int size = blockDim.x;
  unsigned int x_i = keys[i];          // key of integer at position i
  double val = values[i];
  unsigned int p_i = (x_i >> bit) & 1;  // least significant bit

  // replace keys with least significant bits
  keys[i] = p_i;
  __syncthreads();

  // inclusive sum scan to calc the number of 1's up to and including keys[i]
  unsigned int True_before = scanHillisSteele(keys);
  unsigned int True_total  = keys[size-1];     // total number of True bits
  unsigned int False_total = size - True_total;  // total number of False bits
  __syncthreads();

  // now, x_i needs to be put in the right position (has to be stable sort)
  if (p_i) {  // bit is a 1
    keys[False_total + True_before - 1] = x_i;
    values[False_total + True_before - 1] = val;
  } else {    // bit is a 0
    keys[i - True_before] = x_i;
    values[i - True_before] = val;
  }
  __syncthreads();
}

void radix_sort(unsigned int *keys, double * values, const int ARRAY_SIZE) {
  for (int bit = 0; bit < 32; ++bit) {
    partition_by_bit<<<1, ARRAY_SIZE>>>(keys, values, bit);
  }
}

int main() {
  const int ARRAY_SIZE = 15;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned int);

  // generate the input array on the host
  unsigned int h_keys[ARRAY_SIZE]{
    48, 30, 10, 72, 47, 23, 98, 89, 29, 35, 97, 91, 33, 28, 41};
  double h_values[ARRAY_SIZE]{
    48.8, 30.0, 10.0, 72.2, 47.7, 23.3, 98.9, 89.9, 29.9, 35.5, 97.7, 91.1, 33.3, 28.8, 41.1};
  
  // declare GPU memory pointers
  unsigned int * d_keys;
  double * d_values;

  // allocate GPU memory
  cudaMalloc((void **) &d_keys, ARRAY_BYTES);
  cudaMalloc((void **) &d_values, ARRAY_SIZE * sizeof(double));

  // transfer the array to the GPU
  cudaMemcpy(d_keys, h_keys, ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, h_values, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);

  // launch the kernel
  radix_sort(d_keys, d_values, ARRAY_SIZE);
  cudaDeviceSynchronize();
  
  // transfer the resulting array to the cpu
  cudaMemcpy(h_keys, d_keys, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_values, d_values, ARRAY_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "keys:" << std::endl;
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    std::cout << h_keys[i] << " " << std::flush;
  }
  std::cout << std::endl << "values:" << std::endl;
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    std::cout << h_values[i] << " " << std::flush;
  }
  std::cout << std::endl;

  // free GPU memory allocation
  cudaFree(d_keys);
  cudaFree(d_values);

  return 0;
}

// http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci360/lecture_notes/radix_sort_cuda.cc
