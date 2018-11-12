# My solution to homework assignment 3

### Compile
```
mkdir build
cd build
cmake ..
make
```
### Usage
`./hw3 example.jpg`

I decided to use the *Hillis & Steele* algorithm for the *scan* operation. With O(n log n) it is less work efficient than serial or the *Blelloch* algorithm. However, it takes half as many steps as *Blelloch* (log(n) vs 2*log(n)). In this simple homework example we certainly have more processors than work and therefore prefer a step efficient solution: since there are plenty of processors we are willing to do extra work to save some steps.

**The scan implementation is taken from 
[here (slide 10)](http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf) or [here (Example 39-1)](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html). Note however, that the code from these sources does not work and the following line has to be changed from**

`temp[pout*n+thid] += temp[pin*n+thid - offset];`

**to**

`temp[pout*n+idx] = temp[pin*n+idx - offset] + temp[pin*n+idx];`

**for the *scan* to work properly.**

The histogram is currently calculated using *atomics*. This is certainly a bottleneck. A better way to do this would be to give each thread a set of items to bin and a local copy of the histogram to bin into. That way we do not need atomic add!

Then use the reduction technique within a block to combine the n_threads histograms in that block into one single histogram.

Then use a single thread in that block to add the per-block histogram to the global histogram using atomic add.