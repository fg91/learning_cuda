// "Copyright 2018 <Fabio M. Graetz>"
#include <stdlib.h>
#include <iostream>
#include "./utils.h"
#include "./blur.h"
#include "./reference_calc.h"
#include "./gputimer.h"

int main(int argc, char **argv) {
  // width and height of block for launching kernels
  // my GPU supports a maximum of 1024 threads per block
  int size = 32;
  
  // pointers for input and output image on host and device
  uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;

  // input and output filenames
  string input_file;
  string output_file;

  switch (argc) {
    case 2:
      input_file = string(argv[1]);
      output_file = "blurred.png";
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
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA,
             &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

  GpuTimer timer;
  timer.Start();
  gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
                numRows(), numCols(), d_redBlurred, d_greenBlurred,
                d_blueBlurred, filterWidth);
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  cout << "Your code ran in: " << timer.Elapsed() << "msecs." << endl;
  
  // copies result back to host
  size_t numPixels = numRows() * numCols();
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyDeviceToHost));
  // outputs blurred image
  postProcess(output_file, h_outputImageRGBA);
  // creates and outputs a reference blurredimage on CPU
  referenceCalculation(h_inputImageRGBA, h_outputImageRGBA, numRows(), numCols(),
  h_filter, filterWidth);
  postProcess("ref.png", h_outputImageRGBA);

  // frees memory on GPU
  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));
  cleanup();
  cleanupCu();
}
