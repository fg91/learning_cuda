/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/*
  https://nvlabs.github.io/cub/example_block_scan_8cu-example.html
*/

/******************************************************************************
 * Simple demonstration of cub::BlockScan
 *
 * Example compilation string:
 *
 * nvcc example_block_scan_sum.cu -gencode=arch=compute_20,code=\"sm_20,compute_20\" -o example_block_scan_sum
 * nvcc -o test -I=/home/XX/cub example_block_scan_cub.cu
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <stdio.h>
#include <iostream>

#include <cub/cub.cuh>

using namespace cub;
using std::cout;
using std::endl;
using std::flush;
using std::string;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose      = false;
int g_iterations    = 100;

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm  ALGORITHM>
__global__ void BlockPrefixSumKernel(int *d_in, int *d_out, clock_t *d_elapsed) {

  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
  // Specialize BlockScan type for our thread block
  typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;

  } temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, data);

  __syncthreads();

  // Start cycle timer
  clock_t start = clock();

  // Compute scan
  int aggregate;
  BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

  // Stop cycle timer
  clock_t stop = clock();

  __syncthreads();

  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0) {
    *d_elapsed = (start > stop) ? start - stop : stop - start;
    d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
  }
}

//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/*
 * Initialize exclusive prefix sum problem (and solution).
 * Returns the aggregate
 */

int Initialize(int *h_in, int *h_ref, int num_elemnts) {
  int inclusive = 0;

  for (int i = 0; i < num_elemnts; i++) {
    h_in[i] = i % 17;
    h_ref[i] = inclusive;
    inclusive += h_in[i];
  }

  return inclusive;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm  ALGORITHM>
void Test() {
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Alloc host arrays and initialize test case
  int *h_in  = new int[TILE_SIZE];
  int *h_ref = new int[TILE_SIZE];
  int *h_gpu = new int[TILE_SIZE + 1];

  int h_aggregate = Initialize(h_in, h_ref, TILE_SIZE);

  // Initialize device arrays
  int *d_in  = nullptr;
  int *d_out = nullptr;
  clock_t *d_elapsed = nullptr;

  cudaMalloc(&d_in,      sizeof(int) * TILE_SIZE);
  cudaMalloc(&d_out,     sizeof(int) * (TILE_SIZE + 1));
  cudaMalloc(&d_elapsed, sizeof(clock_t));

  // Display input problem data
  if (g_verbose) {
    cout << "Input data: " << flush;
    for (int i = 0; i < TILE_SIZE; i++) {
      cout << h_in[i] << " " << flush;
    }
    cout << endl;

  }

  // Transfer problem data to device
  cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

  string alg = (ALGORITHM == BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING" : (ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ? "BLOCK_SCAN_RAKING_MEMOIZE" : "BLOCK_SCAN_WARP_SCANS";

  cout << "BlockScan " << TILE_SIZE << " items (" << BLOCK_THREADS << " threads,\t"
       << ITEMS_PER_THREAD << " items per thread)\t"
       << "using algorithm " << alg << ":\t" << flush;
  
  // Run kernel several times and average performance
  clock_t elapsed_scan_clocks = 0;
  for (int i = 0; i < g_iterations; ++i) {
    // Run scan kernel
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<1, BLOCK_THREADS>>>(d_in,
											   d_out,
											   d_elapsed);

    // copy results from device
    clock_t scan_clocks;
    cudaMemcpy(h_gpu, d_out, sizeof(int) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(&scan_clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost);
    elapsed_scan_clocks += scan_clocks;
  }

  // check scanned items
  bool correct = 1;
  for (int i = 0; i < TILE_SIZE; ++i) {
    if (h_gpu[i] != h_ref[i]) {
      cout << "Incorrect result @ offset " << i << "(" << h_gpu[i] << "!=" << h_ref[i] << ")" << endl;
      correct = false;
      break;
    }
  }

  // check total aggregate
  if (h_gpu[TILE_SIZE] != h_aggregate) {
    cout << "Incorrect aggregate" << endl;
    correct = false;
  }
  if (correct) cout << "Correct, " << flush;

  // Display timing results
  cout << "Average clocks per 32-bit int scanned: " << float(elapsed_scan_clocks) / TILE_SIZE / g_iterations << endl;

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_ref) delete[] h_ref;
  if (h_gpu) delete[] h_gpu;

  if (d_in) cudaFree(d_in);
  if (d_out) cudaFree(d_out);
  if (d_elapsed) cudaFree(d_elapsed);
}

int main(int argc, char *argv[]) {
  // display GPU name
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  cout << " Using device " << props.name << endl;

  // Run test cases
  Test<1024, 1, BLOCK_SCAN_RAKING>();
  Test<512, 2, BLOCK_SCAN_RAKING>();
  Test<256, 4, BLOCK_SCAN_RAKING>();
  Test<128, 8, BLOCK_SCAN_RAKING>();
  Test<64, 16, BLOCK_SCAN_RAKING>();
  Test<32, 32, BLOCK_SCAN_RAKING>();

  Test<1024, 1, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<512, 2, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<256, 4, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<128, 8, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<64, 16, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<32, 32, BLOCK_SCAN_RAKING_MEMOIZE>();

  Test<1024, 1, BLOCK_SCAN_WARP_SCANS>();
  Test<512, 2, BLOCK_SCAN_WARP_SCANS>();
  Test<256, 4, BLOCK_SCAN_WARP_SCANS>();
  Test<128, 8, BLOCK_SCAN_WARP_SCANS>();
  Test<64, 16, BLOCK_SCAN_WARP_SCANS>();
  Test<32, 32, BLOCK_SCAN_WARP_SCANS>();
  
  return 0;
}
