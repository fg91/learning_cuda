# learning_cuda
Notes and Homework assignments to the course "Intro to Parallel Programming" by Nvidia on Udacity

## CPU vs GPU
**CPU** Complex control hardware, optimized for *latency*, trying to optimize the time elapsed on one particular task
**Pro**: Flexibility + Performance!
**Con**: Expensive in terms of Power

**GPU** Simpler control hardware, optimized for *throughput*
**Pro**: More hardware for computation and potentially more power efficient (Ops/Watt)
**Con**: More restrictive programming models

## Cuda program
Same code programs the host (CPU) and the device (GPU), both having their own dedicated memory. The host controls the device. However, this is currently changing. Modern GPUs can launch their own kernels.

**Typical Operations**:
1. Data from CPU to GPU or vice versa (cudaMemcpy)
1. Allocate GPU memory (cudaMalloc)
1. Invoking programs on the GPU: Host launches kernels on the device

## A typical GPU program
1. CPU allocates memory on GPU (cudaMalloc)
1. CPU copies input data from CPU to GPU (cudaMemcpy)
1. CPU launches kernel(s) on GPU to process the data (kernel launch)
1. CPU copies results bacj to CPU from GPU (cudaMemcpy)

Data transfer might be expensive, minimize it where possible and maximize computation to communication ratio.

## Big Idea
Kernels look like serial programs. Write your program as if it will run in *one* thread. The GPU will run that program on *many* threads.


The GPU is good at:
1. Efficient launching lots of threads
1. Running lots of threads in parallel

