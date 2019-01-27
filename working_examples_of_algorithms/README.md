# My working examples for the main algorithms discussed in the course
## Reduce
## Scan
### Hillis & Steele
With O(n log n) *Hillis & Steele scan* is less work efficient than serial or the *Blelloch* algorithm. However, it takes half as many steps as *Blelloch* (log(n) vs 2*log(n)). When there are plenty of processors we are willing to do extra work to save some steps.

**The scan implementation is taken from 
[here (slide 10)](http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf) or [here (Example 39-1)](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html). Note however, that the code from these sources does not work and the following line has to be changed from**

`temp[pout*n+thid] += temp[pin*n+thid - offset];`

**to**

`temp[pout*n+idx] = temp[pin*n+idx - offset] + temp[pin*n+idx];`

**for the *scan* to work properly.**

`scanHillisSteele_without_double_buffer.cu` works without double buffer and can scan arrays that are larger than 1024 elements because it can handle mutliple threadblocks.

## Sort
### Radix sort