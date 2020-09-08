import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import numpy as np
from CusClas import TurtleKMeans

def generate_points():

    l1 = 10
    l2 = 20
    points = []
    for i in range(l1):

        angle = 2. * 3.14159365 * float(i) / l1
        radius = float(i**2)
        center1 = np.array([radius*math.cos(angle), 2*radius*math.sin(angle)])
        center2 = np.array([2*radius*math.cos(angle), 5*radius*math.sin(angle)])
        center3 = np.array([5*radius*math.cos(angle), 9*radius*math.sin(angle)])
        for j in range(l2):
            if np.random.uniform() < 0.5:
                points.append(center1 + np.random.normal(0., 10., size=2))
            if np.random.uniform() < 0.25:
                points.append(center2 + np.random.uniform(0., 20., size=2))
            if np.random.uniform() < 0.10:
                points.append(center3 + np.random.uniform(0., 50., size=2))

    points = np.array(points)

    return points

def plot_points(fig, km, center_std, size_std, count_std, empty_occur):

    plt.clf()
    plt.tight_layout()
    gs = GridSpec(2, 4)

    ax_clus = fig.add_subplot(gs[:2, :2])

    ax_center = fig.add_subplot(gs[0, 2])
    ax_size = fig.add_subplot(gs[0, 3])
    ax_count = fig.add_subplot(gs[1, 2])
    ax_empty = fig.add_subplot(gs[1, 3])

    ax_clus.scatter(new_points[:, 0], new_points[:, 1], c="r", s=10, marker="o")
    ax_clus.scatter(km.cluster_centers[-1, :, 0], km.cluster_centers[-1, :, 1], c="g", s=100, marker="x")
    ax_clus.set_title("points and centers")

    ax_center.plot(np.arange(len(center_std)), center_std, c="black")
    ax_center.set_title("center_std")
    ax_size.plot(np.arange(len(size_std)), size_std, c="black")
    ax_size.set_title("size_std")
    ax_count.plot(np.arange(len(count_std)), count_std, c="black")
    ax_count.set_title("count_std")
    ax_empty.plot(np.arange(len(empty_occur)), empty_occur, c="black")
    ax_empty.set_title("empty_occur")

    plt.show()

lr = 0.01
empty_lr = 0.001

km = TurtleKMeans(30, 100)
new_points = generate_points()
km.partial_fit(new_points, lr, empty_lr)
km.cluster_centers += np.random.uniform(0., 150., size=km.cluster_centers.shape)

center_std = list()
size_std = list()
count_std = list()
empty_occur = list()

fig = plt.figure()

for i in range(3000):

    new_points = generate_points()
    km.partial_fit(new_points, lr, empty_lr)

    center_std.append(math.log(km.mean_cluster_center_std() / lr))
    size_std.append(km.mean_cluster_size_std())
    count_std.append(km.mean_cluster_count_std())
    empty_occur.append(km.empty_cluster_occurrence())


    if i % 100 == 0:

        print("iter: " + str(i) + 
              "\tcenter_std: " + str(center_std[-1]) + 
              "\tsize_std: " + str(size_std[-1]) + 
              "\tcount_std: " + str(count_std[-1]) +
              "\tempty_occur: " + str(empty_occur[-1]))

str_center_std = "\n".join([str(i) + "\t" + str(value) for i, value in enumerate(center_std)])
str_size_std = "\n".join([str(i) + "\t" + str(value) for i, value in enumerate(size_std)])
str_count_std = "\n".join([str(i) + "\t" + str(value) for i, value in enumerate(count_std)])
str_empty_occur = "\n".join([str(i) + "\t" + str(value) for i, value in enumerate(empty_occur)])
        
with open(".\center_std.txt", "w+") as f:
    f.write(str_center_std)

with open(".\size_std.txt", "w+") as f:
    f.write(str_size_std)
    
with open(".\count_std.txt", "w+") as f:
    f.write(str_count_std)
    
with open(".\empty_occur.txt", "w+") as f:
    f.write(str_empty_occur)
        
plot_points(fig, km, center_std, size_std, count_std, empty_occur)
