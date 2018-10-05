// "Copyright 2018 <Fabio M. Graetz>"
#ifndef UTILS_H_
#define UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

using std::cerr;
using std::endl;

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func,
           const char* const file, const int line) {
  if (err != cudaSuccess) {
    cerr << "CUDA error at: " <<  file << ":" << line << endl;
    cerr << cudaGetErrorString(err) << " " <<  func << endl;
  }
}
#endif  // UTILS_H_
