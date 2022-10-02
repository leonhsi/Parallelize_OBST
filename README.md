# Parallelize_OBST

build program : 
```
make
```

How to run : 
```
./obst --node <INT>
```
`INT` is the number of nodes in a binary search tree, the frequency and failure search cost for each node is generated randomly.

The final result of each parallel method will be verified with serial version

## Method

Implement 3 different parallel methods to speedup optimal binary search tree.


### Layer 1

* Assign one thread to each entry
* Each thread has to compute every node's search cost and find the minimum one.
![](https://i.imgur.com/3GMKpdW.png)


### Layer 2

* Assign one thread to each node in every entry
* Each thread has to compute only one node's search cost 
* Use extra global memory space to store every node's search cost in one entry and find the minimun value for that entry

![](https://i.imgur.com/tXejmgU.png)


### Dynamic Parallelism

* Assign one thread to each entry, and this thread will invoke a child kernel
* Each child kernel has to compute the minimum search cost for this entry parallelly

![](https://i.imgur.com/HhnbXZd.png)

## Result
![](https://imgur.com/xAqbPQ6.png)


