# Lesson 2

## Communication patterns
Threads sometimes need to
1. read from the same input location
1. write to the same output location
1. exchange partial results

### Map
**One-to-one**
There is a 1 to 1 correspondance between input and ouput, each task reads from and writes to a specific place in memory.

Example:
`out[i] = pi * in[i]`

### Scatter
**One-to-many**
Each tasks takes an input and calculates the position in the output(s) by itself, for example sorting.

Example:
```
if (i % 2):
   out[i - 1] += pi*in[i];
   out[i + 1] += pi*in[i];
```

### Gather
**Many-to-one**
Examples:
1. Running averages. Each thread reads values from n locations in memory, averages them and writes the result to another location
1. Blur an image: Set each pixel to the average from the neighbouring pixels (stencil)

Example:
`out[i] = (in[i] + in[i - 1] + in[i + 1]) /3.0f;`

### Stencil
**Several in the neighbourhood to one**
Tasks read input from a fixed neighbourhood (stencil) in an array, for example blurring, convolutions,...

Same example as *Gather* if you do it on every element in an array.

### Transpose
**One-to-one: Tasks re-order data elements in memory, always 1 to 1 correspondance**
When you transpose an array, the elements of the array or the pixels of an image are actually laid out one row at a time and have to be written to someplace scattered in memory according to the stride of this transpose.
You could also formulate this as a gather operation, first gathering the input for the first field in the transposed array, then the input for the second field in the transposed array that is stored in a completely different location (not next to the first in the original array).

Example
```
struct foo {
   float f;
   int i;
}
```
If you have an array of these, the array of structures (AOS) is stored in memory like this:

*fifififi*

If you do a lot of processing on the floats, it might be more efficient to acces all the floats contiguosly by converting this to a structure of arrays (SOA) using a transpose operation:

*fffiii*

Example: Convert i major order to j major order
`out[i + j*128] = out[j + i*128]`

## Question we need to answer:
### 1. How can threads efficiently access memory in conert?
### 2. How to exploit data reuse?
### 3. How can threads communicate partial results safely by sharing memory?

## GPU architecture
A GPU consists of many *Streaming Multiprocessors (SMs)*. Different GPUs have different numbers of SMs (1 to maybe 16 or more).

**Important: The GPU is responsible for allocating the blocks to the SMs**.
As a programmer, you worry about giving the GPU the thread blocks, the GPU worries about how to assign them to the hardware SMs.

Answers to Quiz:
1. A thread block contains many threads
1. An SM may run more than one block
1. A block can only run on one SM
1. All threads in a thread block may cooperate to solve a sub problem
1. All threads that run an a given SM may NOT cooperate to solve a subproblem, only the ones in one single threadblock
1. The programmer can neither specify that two blocks X and Y are run at the same time or after each other
1. The programmer cannot specify on wich SM Block X will run

Because the code is independant of the number of SMs on the GPU, the code will scale forwards to larger and larger GPUs but also to smaller ones as found on tablets or cell phones.

```
__global__ void hello() {
  printf("Hello world!; I'm a thread in block %d\n", blockIdx.x);
}

.
.
.

hello<<<16, 1>>>();
```

This has 16! possible different output because each block of code has a different idx, thus output, but the *order is random*.

### Things CUDA guarantees:
1. All threads in a block run on the same SM at the same time
1. All blocks in a kernel finish before any blocks from the next kernel run

## Memory
Every thread has access to three kinds of memory on the GPU:
1. Local memory that belongs only to the thread
1. Shared memory. Shared between threads of one block. Small amount of memory that sits on the SM directly
1. Global memory. Accessible by threads everywhere

Data is passed by the CPU from the host/CPU memory to the device/GPU memory (global) before launching kernels