# Lesson 5 - Optimizing GPU Programs

## Principles of efficient GPU programming

1. Maximize arithmetic intensity (amount of math operations we do for the amount of time we spend on memory access)
2. Minimize time spent on memory operations
3. *Coalesce* global memory access
4. Avoid thread divergence
5. Move frequently accessed memory to the fast shared memory

## Levels of optimization

1. Picking good algorithms
2. Basic prinicples for efficiency
3. Architecture-specific detailed optimizations
4. "Bit-twiddling" micro-optimization at the instruction level

Examples for those levels on **CPU**:

1. Use mergesort O(n log n) instead of insertion sort O(n^2)
2. Write cache-aware code, i.e. traverse rows vs cols
3. Block for the L1 cache; vector registers SSE, AVX (?) These types of optimization are a *lot* more important in CPU programming compared to GPU programming.

Examples for those levels on **GPU**:

1. On a CPU heap-sort often is slightly faster than merge-sort (both O(n log n)). On the GPU the heap is difficult to parallelize => Pick fundamentally parallel algorithms!
2. **Coalesce** global memory access and use **shared** memory.
3. Optimizing *bank conflicts* in shared memory and optimizing *registers*.

## APOD - Systematic optimization process

-> Analyse -> Parallelize -> Optimize -> Deploy -> (get real life feedback)

1. Analyze: Where can the application benefit from parallelism and by how much.
2. Parallelize: Pick an approach (existing libraries, directives such as OpenMP and OpenACC, programming languages for GPU). Then, pick the right algorithm
3. Profile-driven optimization (measure the performance, adapt the algorithm until you have something that performs well)
4. Deploy early and frequently: Don't optimize in a vacuum. Depley it and get real life feedback. Even 2 or 4 times speedup is useful to customers, don't wait until you have 20 times speedup on something that might not be a bottleneck in real life.

## Weak vs strong Scaling

1. **Weak scaling:** Run a larger problem (or more problems) in same time
2. **Strong scaling:** Run a problem faster (at the same size)

## Understanding Hotspots
Don't rely in intuiton!

Run a profiler:

1. gprop
2. vtune
3. VerySleepy

### Andahls Law
Total speedup from parallelization is limited by portion of time spent doing some thing to be parallelized

max speedup = 1 / (1 - p), where p is % of parallelizable time

Example: When 50% of time is spent on things that are parallelizable, the max speedup is x2!

Remeber: Often, when a hotspot is ported to GPU, it is so much faster that it is no longer a bottleneck and you have to look at other hotspots that are now more important than further optimizing the first hotspot.

## Parallelize
### Example: Matrix transpose
Goal: Switch every element (i,j) with element (j,i).

Matrix is layed out in row major order:

![](pictures/screenshot1.png)

Serial implementation:

```
void transposeCPU(float in[], float out[]) {
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			out[j + i * N] = [i + j * N];  // row major order  		
		}	
	}
}
```

The file `transpose.cu` contains several parallel implementations:

1. **Serial:** 1 thread on GPU transposes a 1024x1024 matrix in 144.368 ms (0.024% of theoretical peakbandwidth). This is a long time but might be completely fine if done only once in the code or on a very small matrix.
2. **Parallel per row:** One thread per row, 1.23155 ms (2.8% of theoretical peak bandwidth)
3. **Parallel per element:** One thread per element, in a 32x32 grid of 32x32 thread blocks. 0.067168 ms (52% of theoretical peak bandwidth)

At this point, is there something else we can do? Usually you can optimize the compute and the memory access. Transposing costs almost no compute, so let's optimize memory access:

From `deviceQuery.cpp` find out the memory clock:

* Memory Clock rate:                             5505 Mhz (clocks/s)
* Memory Bus Width:                              352-bit = 44 bytes/clock

The theoretical peak bandwidth therefore is 242 GB/s!

Good rule of thumb:

Using:

* 40-60% of memory bandwidth => okay
* 60-75% of memory bandwidth => good
* > 75% of memory bandwidth => excellent

So how well is our code doing?

Reading and writing 1024x1024 elemets of 4 bytes in 0.067168 ms gives a bandwidth of 125 GB/s. This is 51% of theoretical peak bandwidth.

So we can probably do better!

Whenever you see really low DRAM utilization or really low percentage bandwidth, your first guess should be **coalescing**.

The GPU is always accessing DRAM (global memory) in pretty large chunks, 32 or 128 bytes at a time.

We need **the fewest total memory transactions when threads in a warp access contiguous adjacent memory locations**.

![](pictures/screenshot2.png)

**Left:** Example of good coalescing. Every thread either reading or writing an adjacent memory location. 

**Middle:** Random access pattern: Threads in the warp are reading or writing completely random memory locations. You get poor coalescing. If accesses are spread out all over memory the total number of chunks of memory that we have to read (vertical black lines) could be as large as the number of threads in the warp.

**Right:** Strided access pattern: Threads access memory location that is a function of their thread Id times some stride. Strided accesses range from being "okay" to "bad". Stride 2 doubles the number of memory transactions (halfing the quality of the coalescing). If the stride between elements is large enough every thread in the warp is accessing a different 32 or 128 byte chunk of memory => bad behaviour.

Let's look at our example kernel *Parallel per element*:

```
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

// one thread per element
out[j + i * N] = xin[i + j * N];
```

**Reading:** Threads with adjacent thread idx x are reading adjacent values of the input matrix. This is exactly what we want, this is good coalescing.

**Writing:** Threads with adjacent values of i are writing to places separated in memory by N. This is bad coalescing and the root of our problem.

**Most good GPU codes are memory limited**

So, always start by measuring your achieved bandwidth to see if you are using memory efficiently. If not, ask yourself why not!

Calculating the memory bandwidth by hand is nice but there are tools that do this for us. Use nSignt Eclipse on Mac or Linux or `nvvp` (Nvidia visual profiler). Look at *Global memory load efficiency* and *global memory store efficiency*.

The numbers mean: Of all the bytes that we fetched with each memory transaction, how many of them are actually useful?

The results for our latest kernel are 100% (fully coalesce access) and 12.5%, respectively.

We need to achieve a higher bandwidth while writing the output to the matrix.
