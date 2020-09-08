# Turtle KMeans algorithm

## Introduction
This is an implementation of the classic K-means clustering algorithm, but during an update the cluster-centers are re-positioned only by a small amount instead of the direct re-positioning to the new cluster-centroids. This allows the cluster-centers to move more slowly (like a turtle, hence the name) in contrast to the classic case, making this algorithm more robust against noise. The points to be clustered can be added in batches during the training. Temporarily empty clusters are moved towards close, larger clusters, again, by a small amount. This algorithm is particularly effective, when the points to be clustered are too numerous and cannot fit into the memory all at once, or the point-set is updated incrementally.

## Framework
The algorithm is implemented and tested in Python 3.8 using numpy 1.18.5.

## How it works

### Initialization
The initialization is similar to the K-means++ algorithm.

### Cluster update
In the classic K-means algorithm each new cluster-center is calculated by the centroid of the points that belong to the corresponding cluster. This code uses an interpolation between the old and the new centers instead. The interpolation coefficient is the so called "learning rate" (in the code *lr*) and the new center is calculated by *new_center* = (1 - *lr*) \* *old_center* + *lr* \* *centroid*. Also, the centers of the empty clusters are moved towards randomly chosen clusters during an update in a similar manner, but with a different, so called empty learning rate (in the code *empty_lr*): *new_center* = (1 - *empty_lr*) \* *old_center* + *empty_lr* \* *random_center*. It is advised to keep the empty learning rate smaller than the (normal) learning rate. The random centers are sampled so, that each center has a probability *p* to be chosen, where *p* is proportional to *S*/*D*, where *S* is the size of the cluster and *D* is the distance between the random cluster and the empty cluster. The size of a cluster is defined by the mean L2-distance of the cluster-center and the cluster-points.

### "Memory"
The cluster-centers, cluster-counts, and cluster-sizes are stored after each update on a batch, until a certain number of batches (*memory_size*) is reached. Then, the oldest observations start to be erased. Thus, the shape of these numpy arrays are the following:
- cluster-centers: (*memory_size*, *K*, *features*)
- cluster-counts: (*memory_size*, *K*)
- cluster-sizes: (*memory_size*, *K*)

This memory feature can be used to observe the convergence of the algorithm: the functions ```mean_cluster_center_std```, ```mean_cluster_size_std```, ```mean_cluster_count_std``` and ```empty_cluster_occurrence``` all serve the purpose of determining the convergence, i.e. the clustering endpoint.

### How to use
The file ```test.py``` serves as a basic example for the usage of this algorithm. First, we have to import the TurtleKMeans class:

```from CusClas import TurtleKMeans```

Next, we create a TurtleKMeans instance with a specific *K* and *memory_size*:

```km = TurtleKMeans(K, memory_size)```

And this is how we update the instance using a *point_batch* and learning rates *lr* and *empty_lr*:

```km.partial_fit(point_batch, lr, empty_lr)```
