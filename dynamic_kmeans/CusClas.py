import numpy as np

class TurtleKMeans():

    def __init__(self, K, memory_size):

        self.K = K
        self.memory_size = memory_size
        self.cluster_centers = None  # shape = (memory_size, K, features)
        self.cluster_counts = None  # shape = (memory_size, K)
        self.cluster_sizes = None  # shape = (memory_size, K)

        pass

    def kmpp_init(self, points):

        distances = None
        current_centers = None
        for i in range(self.K):

            if i == 0:

                random_index = np.random.choice(points.shape[0])
                current_centers = np.expand_dims(points[random_index], axis=0)  # shape = (1, f)

                distances = np.expand_dims(points, axis=1) - np.expand_dims(current_centers, axis=0)  # shape = (P, 1, f)
                distances = np.sqrt(np.sum(np.square(distances), axis=2))  # shape = (P, 1)

            else:

                probs = np.prod(distances, axis=1)
                probs = probs / np.sum(probs, axis=0, keepdims=True)

                random_index = np.random.choice(points.shape[0], p=probs)
                current_centers = np.concatenate([current_centers, 
                                                  np.expand_dims(points[random_index], axis=0)], axis=0)  # shape = (i+1, f)

                new_distances = points - np.expand_dims(points[random_index], axis=0)  # shape = (P, f)
                new_distances = np.sqrt(np.sum(np.square(new_distances), axis=1))  # shape = (P, )
                new_distances = np.expand_dims(new_distances, axis=1)  # shape = (P, 1)
                distances = np.concatenate([distances, new_distances], axis=1)

            self.cluster_centers = np.expand_dims(current_centers, axis=0)

    def partial_fit(self, points, lr, empty_lr):

        # points shape = (P, f)

        if self.cluster_centers is None:

            # KMeans++ like initialization.
            self.kmpp_init(points)
            self.partial_fit(points, lr, empty_lr)

        else:

            current_centers = np.copy(self.cluster_centers[-1])  # shape = (K, f)

            # Generate the cluster - point distances.
            distance_map = np.expand_dims(current_centers, axis=0)  # shape = (1, K, f)
            distance_map = distance_map - np.expand_dims(points, axis=1)  # shape = (P, K, f)
            distance_map = np.sqrt(np.sum(np.square(distance_map), axis=2))  # shape = (P, K)

            # Calculate the cluster index for each point.
            distance_map = np.argmin(distance_map, axis=1)  # shape = (P)

            # Collect the points for each center.
            bag_of_points = [[] for _ in range(self.K)]
            for point_index, cluster_index in enumerate(distance_map):                

                bag_of_points[cluster_index].append(np.copy(points[point_index]))

            bag_of_points = list(map(np.array, bag_of_points))  # a list of shape (cluster_count, f) points

            # Move each non-empty cluster towards the new centers.
            # Also, calculate the cluster counts and cluster sizes.
            current_cluster_sizes = np.zeros(self.K, dtype=float)
            current_cluster_counts = np.zeros(self.K, dtype=int)
            for cluster_index in range(self.K):

                current_cluster_counts[cluster_index] = bag_of_points[cluster_index].shape[0]

                if current_cluster_counts[cluster_index] != 0:

                    new_center = np.mean(bag_of_points[cluster_index], axis=0)

                    current_size = bag_of_points[cluster_index] - np.expand_dims(new_center, axis=0)
                    current_size = np.sqrt(np.sum(np.square(current_size), axis=1))
                    current_size = np.mean(current_size)

                    current_cluster_sizes[cluster_index] = current_size

                    current_centers[cluster_index] = lr * new_center + (1. - lr) * current_centers[cluster_index]
            
            # Move each empty cluster center towards a random attractor cluster.
            for cluster_index in range(self.K):

                if current_cluster_counts[cluster_index] == 0:

                    # A cluster has a high probability of being an attractor cluster if:
                    # 1) It is close to the empty cluster. (distance = D)
                    # 2) It has a large size. (size = S)
                    # So p ~ S / D

                    distances = current_centers - np.expand_dims(current_centers[cluster_index], axis=0)
                    distances = np.sqrt(np.sum(np.square(distances), axis=1))
                    distances[cluster_index] = 1.  # against zero div.

                    probs = current_cluster_sizes / distances
                    probs[cluster_index] = 0.  # against choosing the same cluster
                    probs /= np.sum(probs)

                    attractor_cluster = np.random.choice(self.K, p=probs)
                    current_centers[cluster_index] = empty_lr * current_centers[attractor_cluster] + \
                        (1. - empty_lr) * current_centers[cluster_index]

            # Add the calculated current_centers, current_cluster_sizes, and current_cluster_counts
            # to the memory.
            current_centers = np.expand_dims(current_centers, axis=0)
            current_cluster_sizes = np.expand_dims(current_cluster_sizes, axis=0)
            current_cluster_counts = np.expand_dims(current_cluster_counts, axis=0)

            self.cluster_centers = np.concatenate([self.cluster_centers, current_centers])

            if self.cluster_sizes is None:
                self.cluster_sizes = current_cluster_sizes
                self.cluster_counts = current_cluster_counts
            else:
                self.cluster_sizes = np.concatenate([self.cluster_sizes, current_cluster_sizes])
                self.cluster_counts = np.concatenate([self.cluster_counts, current_cluster_counts])

                if self.cluster_sizes.shape[0] == self.memory_size + 1:
                    self.cluster_centers = self.cluster_centers[1:]
                    self.cluster_sizes = self.cluster_sizes[1:]
                    self.cluster_counts = self.cluster_counts[1:]

    def mean_cluster_center_std(self):

        return np.mean(np.std(self.cluster_centers, axis=0), axis=(0, 1))

    def mean_cluster_size_std(self):

        return np.mean(np.std(self.cluster_sizes, axis=0), axis=0)

    def mean_cluster_count_std(self):

        return np.mean(np.std(self.cluster_counts, axis=0), axis=0)

    def empty_cluster_occurrence(self):

        occurrences = list()
        for counts in self.cluster_counts:
            occurrences.append((counts.shape[0] - np.count_nonzero(counts)) / float(counts.shape[0]))
        return np.mean(occurrences)
