//Udacity HW 4
//Radix Sorting

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "utils.h"

using namespace std;
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
  // Convert raw pointers to thrust::device_ptrs
  thrust::device_ptr<unsigned int> d_inputVals_ptr(d_inputVals);
  thrust::device_ptr<unsigned int> d_inputPos_ptr(d_inputPos);

  // Wrap data in thrust::device_vectors
  thrust::device_vector<unsigned int> d_inputVals_vec(d_inputVals_ptr, d_inputVals_ptr + numElems);
  thrust::device_vector<unsigned int> d_inputPos_vec(d_inputPos_ptr, d_inputPos_ptr + numElems);

  // Sort using thrust::sort_by_key
  thrust::sort_by_key(d_inputVals_vec.begin(), d_inputVals_vec.end(), d_inputPos_vec.begin());

  // Copy sorted values to output pointers
  checkCudaErrors(cudaMemcpy(d_outputVals, thrust::raw_pointer_cast(&d_inputVals_vec[0]),
                             numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, thrust::raw_pointer_cast(&d_inputPos_vec[0]),
                              numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

