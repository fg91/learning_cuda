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

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols);

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const string &filename);

void cleanup();

void postProcess(const string &output_file, unsigned char *data_ptr);

#endif  // UTILS_H_
