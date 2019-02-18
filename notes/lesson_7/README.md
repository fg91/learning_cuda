# Lesson 6 Part B - Parallel Computing Patterns

## BFS 

The problem with the algorithm from the previous lesson is that we visit every edge again and again but set  a depth only once for every vertex. Evey node is touched on every iteration.

It would be much better if we could visit every edge once when it is ready to process.

The CPU algorithm is much more work efficient because it maintains a frontier which marks the boundary between visited nodes and unvisited ones.



### Duane Merrill 2012

![](pictures/screenshot1.png)
(e=edges, v=vertices)

Note that node #2 does not have any outgoing edges, hence no neighbors are listed in array C corresponding to node #2. Also, in array R, both R[2] = 5 and R[3] = 5. This is how we would know, just by reading the data structure, that node #2 has no outgoing edges, and the value C[5] = 4 corresponds to node #3 (meaning that node 3 has an outgoing edge to node #4).

![](pictures/screenshot2.png)

Assume that we are one step into BFS and the frontier is "3" and "1".

1. Step: For every node on the frontier, find its starting index in array C.
2. Calculate the number of neighbours for every node on the frontier using array R: `R[v+1]-R[v]` (gives 3 and 1)
3. Allocate space to store the new frontier. Each node needs to know where in that array it might copy its edge list. Operation is called *allocate*, which is based on scan; we just scan the number of neighbours. In our example we would begin with the array `in = [3,1]` because vertex 1 has 3 neighbours and vertex 3 has 1 neighbour. We scan this array with an exclusive sum scan and get `out = [0,3]`. Node 1 starts to write its neighbours at offset `0`, Node 3 starts to write its neighbours at offset `3`.
4. Copy each block of node edges list to this array.
5. Check in array D for nodes that have already been visited (additional predicate array). This applies to the 0. So the array `[0,2,4,4]` has to be compacted into `[2,4,4]`. The 4 will be visited twice in the next iteration. That is wasteful. Check the paper for solutions to this.

![](pictures/screenshot3.png)

#### Summary
**Big idea:** The basic step is a *sparse copy*. We start with a frontier, look up the adjacency list for the vertices in that frontier. Then we copy them in a contiguous block to make the next frontier. Repeat.

=> linear work, more than 3 billion vertices per second

Initialize: set starting node's depth = 0. Set starting frontier = neighbours of starting node.


2. Frontier: find starting indices for neighbours. How many neighbours?
3. Allocate space for new frontier
4. Copy edge list to new array
5. Cull visited elements (and probably also mark the ones that are visited in this iteration/mark depth).

### Linked List
The linked list / linear chain graph is hard to parallelize.

If we do BFS here it is going to take n steps to get to end of graph.

What algorithm can find the end of the list for each node?

Assume that each node has a *next* pointer (const) and also a *chum* pointer (can be changed, initialized to point to self). At the end of the algorithm we want every *chum* pointer to point to the end of the linked list.

Naive approach: in every iteration, set the chum pointer to chum.next until we reach a node where next is NULL.


![](pictures/screenshot4.png)
![](pictures/screenshot5.png)
![](pictures/screenshot6.png)

On each iteration, the length of the chum pointer is going to be one more than it was on the iteration before.


This gives us n^2 work. N processors working for n steps.

The serial algorithm can solve this problem with a time complexity of O(n).

The goal of parallel computing is to find an algorithm with the same work complexity as the serial algorithm but fewer steps. Sometimes this is not possible but we are willing to do more work if in return we get fewer steps.

The problem here is that we repeat a lot of work the nodes down the chain already did for us.

![](pictures/screenshot7.png)

If we don't jump to chum.next but to chum.chum we only need log(n) steps! The work done is n log(n) but only log(n) steps... Awesome!

```
find_last_node(const int * next, int * chum) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;

	// Initialize
	chum[k] = next[k];
	
	while((chum[k]!=NULL) && (chum[chum[k]!=NULL])) {
		chum[k] = chum[chum[k]];
	}
}
```

The course does not talk about **race condition** in this context. However, it probably is a good idea to use i.e. a double buffer or save `chum[chum[k]]` in a local variable, followd by a `__syncthreads()` barrier, and then write that local variable to `chum[k]`.

## List Ranking
* Input: Each node knows its successor, starting node
* Output: All nodes in order

![](pictures/screenshot8.png)

A serial processor can do this in n steps. How can we do this in fewer steps on a parallel device?

For every node we compute the node that is 2, 4, 8, ..., number of nodes hops away.

![](pictures/screenshot9.png)

Note: when filling i.e. the row *+8* you can do this with two hops from row *0* to row *+4*.

This takes O(n log n) work and O(logn n) steps, thus more work than serial but fewer steps.

In the second phase of the algorithm we do the following:

![](pictures/screenshot10.png)

In the beginning only not starting node 0 is "awake". We launch n threads. Threads for nodes that are "asleep" return immediately.

The immediate successor (+1) for node 0 is five and the index in the result array will be 0 + 1 = 1.

In the next step 0 and 5 are awake. We now calculate the +2 successors for both of them ("awake them").

![](pictures/screenshot11.png)

The +2 successor of 0 is two. Its outpos is 2. The +2 successor of 5 is 7, its outpos is 3.

![](pictures/screenshot12.png)

In the end we use `outpos` to scatter the nodes into to right order.

![](pictures/screenshot13.png)

This second phase takes O(log n) steps as well.

This algorithm is a good example for trading more work for fewer steps.



