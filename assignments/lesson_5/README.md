# Assignment N 5 - Histogramming for Speed

* Kernel `histo_atomic`: 2.854880 msecs. Global memory atomics only.
* Kernel `histo_shared_atomic`: 0.337888 msecs. Creates a separate histogram in shared memory (with shared memory atomics) for each block. Combines those shared memory histograms using global memory atomics.
* Sorting and then reducing by key using thrust: 5.245312 msecs. It is very interesting that this takes longer than the kernels using *atomics*! The course suggested that this approach should be faster than any approach using atomics.

