#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <iostream>

typedef thrust::device_vector<int> int_vec;

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
                                             thrust::plus<int>(),
                                             thrust::not_equal_to<int>());

  // resize histogram to number of unique elements
  hist_values.resize(num_bins);
  hist_counts.resize(num_bins);

  // calculate number of elements per bin
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<int>(1),
                        hist_values.begin(),
                        hist_counts.begin());
}

void histogram(int_vec &data, int_vec &dense_hist) {
  thrust::device_vector<int> sparse_hist_values;
  thrust::device_vector<int> sparse_hist_counts;
  sparse_histogram(data, sparse_hist_values, sparse_hist_counts);
  
  thrust::fill(dense_hist.begin(), dense_hist.end(), 0);

  thrust::scatter(sparse_hist_counts.begin(),
                  sparse_hist_counts.end(),
                  sparse_hist_values.begin(),
                  dense_hist.begin());
}


int main() {
  const int num_bins = 10;

  thrust::host_vector<int> H(10);
  H[0] = 1;
  H[1] = 1;
  H[2] = 3;
  H[3] = 6;
  H[4] = 1;
  H[5] = 1;
  H[6] = 5;
  H[7] = 6;
  H[8] = 7;
  H[9] = 6;

  // Copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;
  int_vec hist(num_bins);

  histogram(D, hist);

  std::cout << "Values:" << std::endl;
  print_vector(D);
  std::cout << "Histogram:" << std::endl;
  print_vector(hist);
  
  return 0;
}


/*
  https://www.youtube.com/watch?v=cGffGYBbtbk
  https://github.com/thrust/thrust/blob/master/examples/histogram.cu
 */
