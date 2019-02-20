#include <stdio.h>
#include "gputimer.h"
#include "utils.h"

const int BLOCKSIZE = 128;
const int NUMBLOCKS = 1000;
const int N = BLOCKSIZE * NUMBLOCKS;

/* 
 * TODO: modify the foo and bar kernels to use tiling: 
 * 		 - copy the input data to shared memory
 *		 - perform the computation there
 *	     - copy the result back to global memory
 *		 - assume thread blocks of 128 threads
 *		 - handle intra-block boundaries correctly
 * You can ignore boundary conditions (we ignore the first 2 and last 2 elements)
 */

__global__ void foo(float out[], float A[], float B[], float C[], float D[], float E[]) {
  // There is no reuse here, tiling makes no sense!
  int i = threadIdx.x + blockIdx.x*blockDim.x; 
  
  out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
}

__global__ void bar(float out[], float in[]) {
  int i = threadIdx.x + blockIdx.x*blockDim.x; 

  out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
}

__global__ void bar_shared(float out[], float in[]) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int idx = threadIdx.x;

  // copy data into shared memory
  extern __shared__ float shared_d_in[];
  shared_d_in[idx + 2] = in[i];

  if (idx == 0) shared_d_in[0] = in[i - 2];
  if (idx == 1) shared_d_in[1] = in[i - 2];
  if (idx == blockDim.x - 2) shared_d_in[idx + 4] = in[i + 2];
  if (idx == blockDim.x - 1) shared_d_in[idx + 4] = in[i + 2];

  __syncthreads();
  
  out[i] = (shared_d_in[idx] + shared_d_in[idx+1] + shared_d_in[idx+2] + shared_d_in[idx+3] + shared_d_in[idx+4]) / 5.0f;
}

void cpuFoo(float out[], float A[], float B[], float C[], float D[], float E[]) {
  for (int i=0; i<N; i++) {
      out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
    }
}

void cpuBar(float out[], float in[]) {
  // ignore the boundaries
  for (int i=2; i<N-2; i++) {
    out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
  }
}

void printArray(float in[], int N) {
  for (int i=0; i<N; i++) { printf("%g ", in[i]); }
  printf("\n");
}

int compareArrays(float *ref, float *test, int N) {
  // ignore the boundaries
  for (int i=2; i<N-2; i++) {
      if (ref[i] != test[i]) {
	  printf("Error: solution does not match reference!\n");
	  printf("first deviation at location %d\n", i);
	  printf("reference array:\n");// printArray(ref, N);
	  printf("solution array:\n");// printArray(test, N);
	  return 1;
	}
    }
  printf("Verified!\n");
  return 0;
}

int main(int argc, char **argv)
{
  // declare and fill input arrays for foo() and bar()
  float fooA[N], fooB[N], fooC[N], fooD[N], fooE[N], barIn[N];
  for (int i=0; i<N; i++) {
      fooA[i] = i; 
      fooB[i] = i+1;
      fooC[i] = i+2;
      fooD[i] = i+3;
      fooE[i] = i+4;
      barIn[i] = 2*i; 
    }
  // device arrays
  int numBytes = N * sizeof(float);
  float *d_fooA;	 	cudaMalloc(&d_fooA, numBytes);
  float *d_fooB; 		cudaMalloc(&d_fooB, numBytes);
  float *d_fooC;	 	cudaMalloc(&d_fooC, numBytes);
  float *d_fooD; 		cudaMalloc(&d_fooD, numBytes);
  float *d_fooE; 		cudaMalloc(&d_fooE, numBytes);
  float *d_barIn; 	cudaMalloc(&d_barIn, numBytes);
  cudaMemcpy(d_fooA, fooA, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fooB, fooB, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fooC, fooC, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fooD, fooD, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fooE, fooE, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_barIn, barIn, numBytes, cudaMemcpyHostToDevice);	
  
  // output arrays for host and device
  float fooOut[N], barOut[N], *d_fooOut, *d_barOut, barOut2[N], *d_barOut2;
  cudaMalloc(&d_fooOut, numBytes);
  cudaMalloc(&d_barOut, numBytes);
  cudaMalloc(&d_barOut2, numBytes);
  
  // declare and compute reference solutions
  float ref_fooOut[N], ref_barOut[N]; 
  cpuFoo(ref_fooOut, fooA, fooB, fooC, fooD, fooE);
  cpuBar(ref_barOut, barIn);
  
  // launch and time foo and bar
  GpuTimer fooTimer, barTimer, barTimer2;
  fooTimer.Start();
  foo<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_fooOut, d_fooA, d_fooB, d_fooC, d_fooD, d_fooE);
  fooTimer.Stop();
  
  barTimer.Start();
  bar<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_barOut, d_barIn);
  barTimer.Stop();

  barTimer2.Start();
  bar_shared<<<N/BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE + 4) * sizeof(float)>>>(d_barOut2, d_barIn);
  barTimer2.Stop();

  
  cudaMemcpy(fooOut, d_fooOut, numBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(barOut2, d_barOut2, numBytes, cudaMemcpyDeviceToHost);
  printf("foo<<<>>>(): %g ms elapsed. Verifying solution...", fooTimer.Elapsed());
  compareArrays(ref_fooOut, fooOut, N);
  printf("bar<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
  compareArrays(ref_barOut, barOut, N);
  printf("bar_shared<<<>>>(): %g ms elapsed. Verifying solution...", barTimer2.Elapsed());
  compareArrays(ref_barOut, barOut2, N);

  checkCudaErrors(cudaFree(d_fooA));
  checkCudaErrors(cudaFree(d_fooB));
  checkCudaErrors(cudaFree(d_fooC));
  checkCudaErrors(cudaFree(d_fooD));
  checkCudaErrors(cudaFree(d_fooE));
  checkCudaErrors(cudaFree(d_barIn));
  checkCudaErrors(cudaFree(d_fooOut));
  checkCudaErrors(cudaFree(d_barOut));
  checkCudaErrors(cudaFree(d_barOut2));
}
