#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <iostream>
#include "gputimer.h"

using std::cout;
using std::endl;

int main() {
  int N = 10 << 25;

  // generate N random numbers
  thrust::host_vector<int> h_vec(N);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the GPU
  thrust::device_vector<int> d_vec = h_vec;

  // sort on device (2.843.595.932 keys per second on GTX 1080 Ti)
  GpuTimer timer;
  timer.Start();
  thrust::sort(d_vec.begin(), d_vec.end());
  timer.Stop();

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  cout << "Thrust sorted " << N << " keys in " << timer.Elapsed() << " ms." << endl;
    
  return 0;
}
