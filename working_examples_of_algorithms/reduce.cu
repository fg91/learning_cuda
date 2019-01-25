#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void sharedMem_reduce_kernel(float *d_out, float *d_in) {
  // shared_data is allocated in the kernel call: 3rd argument
  extern __shared__ float shared_data[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int thrId = threadIdx.x;

  // load shared memory from global memory
  shared_data[thrId] = d_in[myId];
  __syncthreads();

  // do reduction in SHARED memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (thrId < s) {
      shared_data[thrId] += shared_data[thrId + s];
    }
    __syncthreads();
  }

  // thread 0 writes result of this block from shared to global memory
  if (thrId == 0) {
    d_out[blockIdx.x] = shared_data[0];
  }
}

void reduce(float *d_out, float *d_intermediate, float *d_in, int size) {
  // assumption 1: size is not greater than maxThreadsPerBlock ** 2
  // assumption 2: size is a multiple of maxThreadsPerBlock

  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;

  sharedMem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
  
  // now we are down to one block, reduce it
  threads = blocks;  // each block wrote one number into d_intermediate
  blocks = 1;

  sharedMem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
}

int main(int argc, char* argv[]) {

  // --- Checking whether there is a device --- //
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No GPUs found" << std::endl;
    exit(EXIT_FAILURE);
  }
  // --- Properties of device --- //
  int dev = 0;
  cudaSetDevice(dev);

  cudaDeviceProp deviceProps;
  if (cudaGetDeviceProperties(&deviceProps, dev) == 0) {
    std::cout << "Using device " << dev << std::endl;
    std::cout << deviceProps.name << std::endl;
    std::cout << "Global memory: " << deviceProps.totalGlobalMem << std::endl;
    std::cout << "Comoute v: " << static_cast<int>(deviceProps.major) << "." <<
        static_cast<int>(deviceProps.minor)<< std::endl;
    std::cout << "Clock: " << static_cast<int>(deviceProps.clockRate) << std::endl;
  }

  // --- Actual task - Reducing a sequence of numbers wit op "+" --- //
  const int ARRAY_SIZE = 1 << 20;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // generate input array on host
  float h_in[ARRAY_SIZE];
  float sum = 0.0f;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = -1.0f + static_cast<float>(random())/(static_cast<float>(RAND_MAX)/2.0f);
    sum += h_in[i];
  }

  // declare device pointers
  float *d_in, *d_intermediate, *d_out;

  // allocate device memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
  cudaMalloc((void **) &d_out, sizeof(float));

  // transfer input array to device
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);
  cudaEventRecord(stop, 0);

  // calculate elapsed time
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);

  // copy back the result to host
  float h_out;
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Time elapsed: " << elapsed << std::endl;
  std::cout << "Host result: " << sum << ", device result: " << h_out << std::endl;

  cudaFree(d_in);
  cudaFree(d_intermediate);
  cudaFree(d_out);
  return 0;
}
