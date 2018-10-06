// "Copyright 2018 <Fabio M. Graetz>"
#include <stdlib.h>
#include <iostream>
#include "./utils.h"
#include "./rgb2grey.h"

int main(int argc, char **argv) {
  // width and height of block for launching kernels
  // my GPU supports a maximum of 1024 threads per block
  int size = 32;
  
  // pointers for input and output image on host and device
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  // input and output filenames
  string input_file;
  string output_file;

  switch (argc) {
    case 2:
      input_file = string(argv[1]);
      output_file = "gray_scale.png";
      break;
    case 3:
      input_file = string(argv[1]);
      output_file = string(argv[2]);
      break;
    default:
      cerr << "Usage: ./HW1 input_file [output_filename]" << endl;
      exit(1);
  }
  // loads the image and gives us input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage,
             &d_rgbaImage, &d_greyImage,
             input_file);
  // converts image to greyscale on device
  solution_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage,
                         numRows(), numCols(), size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  // copies result back to host
  size_t numPixels = numRows() * numCols();
  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage,
                             sizeof(unsigned char) * numPixels,
                             cudaMemcpyDeviceToHost));
  // outputs greyscale image
  postProcess(output_file, h_greyImage);
  // creates and outputs a reference greyscale image on CPU
  referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());
  postProcess("ref.png", h_greyImage);
  // frees memory on GPU
  cleanup();
}
