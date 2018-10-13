// "Copyright 2018 <Fabio M. Graetz>"
#ifndef UTILS_H_
#define UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using std::cerr;
using std::endl;
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using cv::Mat;

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func,
           const char* const file, const int line) {
  if (err != cudaSuccess) {
    cerr << "CUDA error at: " <<  file << ":" << line << endl;
    cerr << cudaGetErrorString(err) << " " <<  func << endl;
  }
}

size_t numRows();

size_t numCols();

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const string &filename);

void cleanup();

void postProcess(const string &output_file, uchar4 *data_ptr);

#endif  // UTILS_H_
