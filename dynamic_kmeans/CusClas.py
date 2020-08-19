import numpy as np

class TurtleKMeans():

    def __init__(self, K, memory_size):
        '''
        TurtleKMeans constructor.

        :param K: The number of clusters.
        :param memory_size: The size of the iteration history.
        '''

        self.K = K
        self.memory_size = memory_size
        self.cluster_centers = None  # shape = (memory_size, K, features)
        self.cluster_counts = None  # shape = (memory_size, K)
        self.cluster_sizes = None  # shape = (memory_size, K)

        pass

    def kmpp_init(self, points):
        '''
        Initializes the cluster centers, using a kmeans++ like algorithm.

        :param points: The points from which the cluster centers are chosen.
        '''

        # points shape = (P, f)

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
        '''
        Moves the cluster centers towards the centers defined by the actual points.

        If the points are partitioned into batches, but the fitting procedure must be carried
        out on the whole point-set, this function can be used to partially fit the cluster centers
        to the current points. This is achieved by calculating the updated cluster centers using the
        traditional kmeans method, but the new centers are just interpolations between the old and the
        updated centers. The coefficients of the interpolation are determined by the learning rate (lr)
        parameter, so that when lr = 1.0 we get the classical kmeans algorithm. The empty learning rate
        (empty_lr) determines the size of the stochastic movement of the currently empty clusters: each
        empty cluster is moved towards another (attractor) cluster, that is chosen randomly. The closer
        a cluster is to the empty cluster, the higher it's chance is to become an attractor. Also, the
        bigger a cluster is, the higher it's chance is to become an attractor.

        The updated attributes are the cluster_centers, cluster_sizes and cluster_counts. The new centers,
        sizes, and counts are concatenated to these attributes, and if the history buffer exceeds the
        memory_size, the oldest items are deleted.

        :param points: A batch of points that perturb the new cluster centers.
        :param lr: The learning rate, which is the perturbation size of the non-empty clusters.
        :param empty_lr: The empty learning rate, which is the perturbation size of the empty clusters.
        '''

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

            # Collect the point indices for each center.
            bags_of_indices = [[] for _ in range(self.K)]
            for point_index, cluster_index in enumerate(distance_map):
                bags_of_indices[cluster_index].append(point_index)
            bags_of_indices = list(map(lambda x: np.array(x, dtype=np.intp), bags_of_indices))

            # Move each non-empty cluster towards the new centers.
            # Also, calculate the cluster counts and cluster sizes.
            current_cluster_sizes = np.zeros(self.K, dtype=float)
            current_cluster_counts = np.zeros(self.K, dtype=int)
            for cluster_index in range(self.K):

                current_cluster_counts[cluster_index] = len(bags_of_indices[cluster_index])

                if current_cluster_counts[cluster_index] != 0:

                    # Advanced indexing: from the points array take each elements with index i
                    # that is present in bags_of_indices[cluster_index]. These are the points
                    # that currently belong to the cluster with index cluster_index.
                    new_center = np.mean(points[bags_of_indices[cluster_index]], axis=0)

                    current_size = points[bags_of_indices[cluster_index]] - np.expand_dims(new_center, axis=0)
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

    def predict(self, points):
        '''
        Predict the cluster correspondance of a point or multiple points.

        :param points: The point or points we want to make the prediction for.
        :return: The cluster-index correspondance for the point or for each point.
        '''

        # points shape = (P, f) or (f)

        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)
        
        current_centers = np.copy(self.cluster_centers[-1])  # shape = (K, f)
        
        # Generate the cluster - point distances.
        distance_map = np.expand_dims(current_centers, axis=0)  # shape = (1, K, f)
        distance_map = distance_map - np.expand_dims(points, axis=1)  # shape = (P, K, f)
        distance_map = np.sqrt(np.sum(np.square(distance_map), axis=2))  # shape = (P, K)

        # Calculate the cluster index for each point.
        distance_map = np.argmin(distance_map, axis=1)  # shape = (P)

        return distance_map

    def mean_cluster_center_std(self):
        '''
        A convergence indicator.

        Calculates the standard deviation of the coordinates along the history axis
        and averages them.

        :return: The average change in the coordinates.
        '''

        return np.mean(np.std(self.cluster_centers, axis=0), axis=(0, 1))

    def mean_cluster_size_std(self):
        '''
        A convergence indicator.

        Calculates the standard deviation of the cluster sizes along the history axis
        and averages them.

        :return: The average change in the cluster sizes.
        '''

        return np.mean(np.std(self.cluster_sizes, axis=0), axis=0)

    def mean_cluster_count_std(self):
        '''
        A convergence indicator.

        Calculates the standard deviation of the cluster counts along the history axis
        and averages them.

        :return: The average change in the cluster counts.
        '''

        return np.mean(np.std(self.cluster_counts, axis=0), axis=0)

    def empty_cluster_occurrence(self):
        '''
        A convergence indicator.

        Calculates the amount of empty clusters for each history point (in percentage) and
        averages them.

        :return: The average empty cluster occcurrence.
        '''

        occurrences = list()
        for counts in self.cluster_counts:
            occurrences.append((counts.shape[0] - np.count_nonzero(counts)) / float(counts.shape[0]))
        return np.mean(occurrences)
