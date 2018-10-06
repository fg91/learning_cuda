// "Copyright 2018 <Fabio M. Graetz>"
#include "./utils.h"
#include <iostream>
__global__
void rgba_to_greyscale(const uchar4 * const rgbaImage,
                       unsigned char * const greyImage,
                       int numRows, int numCols) {
  // map threadIdx and blockIdx to pixel in image
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col < numCols && row < numRows) {
    int index = numCols * row + col;
    uchar4 pixel = rgbaImage[index];
    // convert to greyscale
    unsigned char grey = static_cast<unsigned char>(.299f * pixel.x + .587f * pixel.y + .114f * pixel.z);
    greyImage[index] = grey;
  }
}

void solution_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
                            uchar4 * const d_rgbaImage,
                            unsigned char * const d_greyImage,
                            size_t numRows, size_t numCols, int blockWidth) {
  int gridX = numCols/blockWidth + 1;
  int gridY = numRows/blockWidth + 1;
  const dim3 blockSize(blockWidth, blockWidth, 1);
  const dim3 gridSize(gridX, gridY, 1);
  // launch kernels
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage,
                                             d_greyImage,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
