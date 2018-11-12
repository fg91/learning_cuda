// "Copyright 2018 <Fabio M. Graetz>"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void scanHillisSteele(int *d_out, int *d_in, int n) {
  int idx = threadIdx.x;
  extern __shared__ int temp[];
  int pout = 0, pin = 1;
  
  temp[idx] = (idx > 0) ? d_in[idx - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    // swap double buffer indices
    pout = 1 - pout;
    pin = 1 - pout;
    if (idx >= offset) {
      temp[pout*n+idx] = temp[pin*n+idx - offset] + temp[pin*n+idx];
    } else {
      temp[pout*n+idx] = temp[pin*n+idx];
    }
    __syncthreads();
  }
  d_out[idx] = temp[pout*n+idx];
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
  scanHillisSteele<<<1, ARRAY_SIZE, 2 * ARRAY_BYTES>>>(d_out, d_in, ARRAY_SIZE);
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
