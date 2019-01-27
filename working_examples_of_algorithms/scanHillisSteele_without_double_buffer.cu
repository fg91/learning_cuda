// "Copyright 2018 <Fabio M. Graetz>"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

template<class T>
__global__ void scanHillisSteele(T *d_out, T *d_in, const int n) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    // extern __shared__ int shared_mem[];
    // shared_mem[i] = i > 0 ? d_in[i-1] : 0;  // gives exclusive sum scan
    
    for (int offset = 1; offset < n; offset <<=1) {
      T temp;
      if (i >= offset) {
        temp = d_in[i - offset];
      }

      __syncthreads();

      if (i >= offset) {
        d_in[i] = temp + d_in[i];
      }

      __syncthreads();
    }
    d_out[i] = d_in[i];
  }
}

int main() {
  const int ARRAY_SIZE = 10;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

  // generate the input array on the host
  int h_in[ARRAY_SIZE]{1, 2, 5, 7, 8, 10, 11, 12, 15, 19};
  int h_out[ARRAY_SIZE];

  // declare GPU memory pointers
  int * d_in;
  int * d_out;

  // allocate GPU memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // transfer the array to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // launch the kernel
  scanHillisSteele<<<3, 4>>>(d_out, d_in, ARRAY_SIZE);
  cudaDeviceSynchronize();
  
  // transfer the resulting array to the cpu
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // print out the input and resulting array
  std::cout << "Input:" << std::endl;
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    std::cout << h_in[i] << " " << std::flush;
  }
  std::cout << std::endl << "Exclusive scan with operation +:" << std::endl;
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    std::cout << h_out[i] << " " << std::flush;
  }
  std::cout << std::endl;

  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

// http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci360/lecture_notes/radix_sort_cuda.cc
