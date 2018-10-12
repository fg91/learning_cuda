/*
   The programmer can not influence the order in which the blocks are run
   Therefore this program has 16! different possible outputs
*/
#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello() {
  printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}

int main() {
  // launch kernels
  hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
  // force the printf()s to flush
  cudaDeviceSynchronize();
  return 0;
}
