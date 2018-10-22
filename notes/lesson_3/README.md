# Lesson 3
## Fundamental GPU algorithms:
1. Reduce
1. Scan
1. Histogram

## Example: digging a hole:

| # workers            | 1 | 4             | 4                                |            |
|----------------------|---|---------------|----------------------------------|------------|
| Time to finish       | 8 | 2             | 4                                | <= "steps" |
| Total amount of work | 8 | 8             | 16                               | <= "work"  |
|                      |   | ideal scaling | the works got in each others way |            |

**There are two types of cost:**
1. Step complexity
1. Work complexity


We say that a parallel algorithm is *work-efficient* if its work complexity is asymtotically the same (within a constant factor) as the work complexity of the sequential algorithm.