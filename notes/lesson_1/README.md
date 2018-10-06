# Lesson 1
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
Kernels look like serial programs. Write your program as if it will run in *one* thread. The GPU will run that program on *many* threads. In the GPU part of the program there is absolutely no degree of parallelism, looks like serial program. The CPU part of the program specifies the degree of parallelism.

**CPU code** functionKernel<<< number of threads >>>(outArray, inArray)


The GPU is good at:
1. Efficient launching lots of threads
1. Running lots of threads in parallel

## Configuring the *kernel launch*
Launch 1 block of 64 threads:

`function<<<1, 64>>>(d_out, d_int)`

1. GPU is capable of running many blocks at the same time
1. Each block has a maximum number of threads that it can support (new GPUs 1024 threads). If you want 128 threads, use `<<<1, 128>>` for instance. If you need 1280 threads use `<<<10, 128>>>` or `<<<5, 256>>>`.
1. Cuda supports 1, 2, or 3 dimensional grids of blocks and blocks of threads: `kernel<<< grid of blocks, block of threads>>>(...)`
1. Syntax: `dim3(x, y, z)` with `dim3(w, 1, 1)==dim3(w)==w`
1. `kernel<<< dim3(bx, by, bz), dim3(tx, ty, tz), shared memory per block in bytes>>>(...)`

Each thread knows the following things
1. Its index within block: `threadIdx.x`, `threadIdx.y`, or `threadIdx.z`
1. Size of a block: `blockDim`
1. Block index, "Which block am I in within the grid": `blickIdx.x`, ...
1. Size of grid: `gridDim`

## Abstraction: Map
`MAP(Elements, Function)`

Takes the following two arguments:
1. Set of elements to process
1. Function to run on each element

## Summary of Lesson 1
1. We can write a program that looks like it runs on one thread
1. We can launch that program on any number of threads
1. Each thread knows its own index in the block + the grid


