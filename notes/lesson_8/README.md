# Lesson 7

## Parallel Optimization Patterns

* Stratton et al. (Optimization and architecture effects in GPU computing and workload performance) InPar 2012. More readable version of same paper in IAAA computer 2012

They analyzed numerous GPU parallelized programs and extracted 7 basic techniques that come up over and over again:

1. **Data layout transformation:** 
	* Reorganize data layout for better global memory performance (coalescing). 
	* Burst utilization, i.e. Aos -> SoA (Array of structurs -> structure of arrays)
	* Partition camping, i.e. *Array of structures of tiled arrays (ASTA)*

	![](pictures/screenshot1.png)
	![](pictures/screenshot2.png)
	![](pictures/screenshot3.png)
2. **Scatter - to - gather transformation:**
	Several input elements give one resulting output element, i.e. for bluring or histograms:
	![](pictures/screenshot4.png)
	In *scatter*, the elements are assigned to input elements and each thread decides where to scatter them. 
	![](pictures/screenshot5.png)
	
	In *gather* the threads are assigned to output elements
	
	![](pictures/screenshot6.png)
	
	![](pictures/screenshot7.png)
	
	The second snippet runs much faster. The first is accessing more memory since it has to read the three output locations to increment them, also there is a race condition and you would need atomic add or syncbarriers to make this work.
	
	* Gatter: many overlapping reads
	* Scatter: many potentially conflicting overlapping writes

3. **Tiling:** Often many threads will have to access overlapping parts of a data structure. Stratton refers to *tiling* as buffering data into fast on-chip memory for repeated access.
	* Either implicit copy from global memory to on-chip memory (cache). Cache-blocking is problematic on GPUs, however, as the amount of cache is limited and the number of threads large. Might limit the number of concurrent threads!
	* Therefore better explicitly allocate fast shared memory

![](pictures/screenshot8.png)
![](pictures/screenshot9.png)
