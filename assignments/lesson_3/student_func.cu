/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]
// exclusive
cdf : [0 4 11]


Your task is to calculate this cumulative distribution by following these
steps.

*/
#include <stdio.h>
#include "utils.h"
#include<device_launch_parameters.h>
#include<device_functions.h>

__global__ void scanHillisSteele(unsigned int *d_in, int n) {
  int idx = threadIdx.x;
  extern __shared__ unsigned int temp[];
  int pout = 0, pin = 1;

  temp[idx] = (idx > 0) ? d_in[idx - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    // swap double buffer indices
    pout = 1 - pout;
    pin = 1 - pout;
    if (idx >= offset) {
      temp[pout*n+idx] = temp[pin*n+idx - offset] + temp[pin*n+idx];
    } else {
      temp[pout*n+idx] = temp[pin*n+idx];
    }
    __syncthreads();
  }
  d_in[idx] = temp[pout*n+idx];
}

__global__
void histogram(const float * const d_in, unsigned int * const d_out, const size_t numBins, float range, float min) {
  int totIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int bin = static_cast<int>((d_in[totIdx] - min) / range * numBins);
  if (bin == numBins) bin = numBins - 1;

  // use atomic add so that the threads don't interfere with each other
  atomicAdd(&d_out[bin], 1);
}

__global__
void getMinMax(const float * const d_in, float * const d_out, bool calcMin) {
  int totIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int thrIdx = threadIdx.x;

  extern __shared__ float shared[];  // allocated with 3rd argument of kernel call

  // copy data into shared memory
  shared[thrIdx] = d_in[totIdx];
  __syncthreads();

  // reduction
  for (unsigned int c = blockDim.x / 2; c > 0; c >>= 1) {
    if (thrIdx < c) {
      if (calcMin)  shared[thrIdx] = min(shared[thrIdx], shared[thrIdx + c]);
      if (!calcMin) shared[thrIdx] = max(shared[thrIdx], shared[thrIdx + c]);
    }
    __syncthreads();
  }

  // thread with index 0 writes result to d_out
  if (thrIdx == 0) {
    d_out[blockIdx.x] = shared[0];
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
  unsigned int* const d_cdf,
  float &min_logLum,
  float &max_logLum,
  const size_t numRows,
  const size_t numCols,
  const size_t numBins) {
  // Here are the steps you need to implement
  // 1) find the minimum and maximum value in the input logLuminance channel
  // store in min_logLum and max_logLum*/
    
  const int threadsPerBlock = 1 << 10;
  const int numBlocks = ceil(static_cast<float>(numRows*numCols) / threadsPerBlock);

  float *d_intermediate, *d_min, *d_max;
  checkCudaErrors(cudaMalloc(&d_intermediate, numBlocks * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));

  // launch reduce kernels
  getMinMax<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_logLuminance, d_intermediate, 0);
  getMinMax<<<1, numBlocks, numBlocks * sizeof(float)>>>(d_intermediate, d_max, 0);  // assumes that numBlocks <= 1024
  getMinMax<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_logLuminance, d_intermediate, 1);
  getMinMax<<<1, numBlocks, numBlocks * sizeof(float)>>>(d_intermediate, d_min, 1);  // assumes that numBlocks <= 1024

  // copy results back to host
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));

  // and clear memory
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));

  // 2) subtract them to find the range
  float range = max_logLum - min_logLum;

  //3) generate a histogram of all the values in the logLuminance channel using
  //   the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int)* numBins));
  histogram <<<numBlocks, threadsPerBlock>>>(d_logLuminance, d_cdf, numBins, range, min_logLum);
  
  //4) Perform an exclusive scan (prefix sum) on the histogram to get
  //   the cumulative distribution of luminance values (this should go in the
  //   incoming d_cdf pointer which already has been allocated for you)      

  scanHillisSteele<<<1, numBins, numBins * 2 * sizeof(unsigned int)>>>(d_cdf, numBins);
}
