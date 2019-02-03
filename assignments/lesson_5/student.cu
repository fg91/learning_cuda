/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <iostream>

#define NUM_BINS 1024
typedef thrust::device_vector<unsigned int> int_vec;

template <typename Vector>
void print_vector(const Vector& v) {
  typedef typename Vector::value_type T;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

void sparse_histogram(int_vec &data,
                      int_vec &hist_values,
                      int_vec &hist_counts) {
  // sort data
  thrust::sort(data.begin(), data.end());

  // number of bins = number of unique values in data (assumes data.size() > 0)
  int num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             1,
                                             thrust::plus<unsigned int>(),
                                             thrust::not_equal_to<unsigned int>());

  // resize histogram to number of unique elements
  hist_values.resize(num_bins);
  hist_counts.resize(num_bins);

  // calculate number of elements per bin
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<unsigned int>(1),
                        hist_values.begin(),
                        hist_counts.begin());
}

void histogram(int_vec &data, int_vec &dense_hist) {
  thrust::device_vector<unsigned int> sparse_hist_values;
  thrust::device_vector<unsigned int> sparse_hist_counts;
  sparse_histogram(data, sparse_hist_values, sparse_hist_counts);
  
  thrust::fill(dense_hist.begin(), dense_hist.end(), 0);

  thrust::scatter(sparse_hist_counts.begin(),
                  sparse_hist_counts.end(),
                  sparse_hist_values.begin(),
                  dense_hist.begin());
}

__global__
void histo_atomic(const unsigned int* const vals,  // INPUT
               unsigned int* const histo,       // OUPUT
               int numVals) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numVals) return;
  atomicAdd(&histo[vals[i]], 1);
}

__global__
void histo_shared_atomic(const unsigned int* const vals,  // INPUT
               unsigned int* const histo,       // OUPUT
               int numVals) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numVals) return;

  // per block histogram in shared memory
  __shared__ unsigned int temp[NUM_BINS];
  temp[threadIdx.x] = 0;
  __syncthreads();

  // add values to per block histogram in shared memory
  atomicAdd(&temp[vals[i]], 1);
  __syncthreads();

  // combine per block histograms in shared memory
  // into final histogram in global memory

  atomicAdd(&histo[threadIdx.x], temp[threadIdx.x]);
}

void computeHistogram(unsigned int* const d_vals,  // INPUT
                      unsigned int* const d_histo,       // OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems) {

  const int threads_per_block = NUM_BINS;
  int num_bytes = sizeof(unsigned int) * NUM_BINS;
  int blocks = ceil(static_cast<float>(numElems)/threads_per_block);

  const int method = 2;

  switch (method) {
    case 0:
      std::cout <<  "Simple histogram using atomics" << std::endl;
      histo_atomic<<<blocks, threads_per_block>>>(d_vals, d_histo, numElems);
      break;

    case 1:
      std::cout << "Histogram using a per-block-histograms in shared memory" << std::endl;
      // For simplicity the code assumes that the number of threads per block
      // is equal to thise number of bins! PAY ATTENTION!
      
      histo_shared_atomic<<<blocks, threads_per_block, num_bytes>>>(d_vals, d_histo, numElems);
      break;

    case 2:
      std::cout << "Histogram by sorting and then reducing by key using thrust" << std::endl;
      // Convert raw pointers to thrust::device_ptrs
      thrust::device_ptr<unsigned int> d_values(d_vals);
      
      // Wrap data in thrust::device_vectors
      thrust::device_vector<unsigned int> d_vals_vec(d_values, d_values + numElems);
      thrust::device_vector<unsigned int> hist(NUM_BINS);
      histogram(d_vals_vec, hist);
      
      // Copy histogram from vector to output pointer
      checkCudaErrors(cudaMemcpy(d_histo, thrust::raw_pointer_cast(&hist[0]),
                                 NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      break;
  }
}
