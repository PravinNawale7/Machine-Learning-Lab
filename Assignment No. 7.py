import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [0.1, 0.6],  
    [0.15, 0.71], 
    [0.08, 0.9],  
    [0.16, 0.85], 
    [0.2, 0.3],   
    [0.25, 0.5],  
    [0.24, 0.1],  
    [0.3, 0.2]    
])
print("Defined points:\n", points)


m1 = np.array([0.1, 0.6])
m2 = np.array([0.3, 0.2])
print("Initial centroids:\nm1:", m1, "\nm2:", m2)


def euclidean_distance(a, b):
    distance = np.sqrt(np.sum((a - b) ** 2))
    print(f"Calculating distance between {a} and {b}: {distance}")
    return distance


clusters = []
for point in points:
    dist_to_m1 = euclidean_distance(point, m1)
    dist_to_m2 = euclidean_distance(point, m2)
    if dist_to_m1 < dist_to_m2:
        clusters.append(1)  
        print(f"Point {point} is closer to m1, assigning to Cluster 1.")
    else:
        clusters.append(2)  
        print(f"Point {point} is closer to m2, assigning to Cluster 2.")


clusters = np.array(clusters)
print("Cluster assignments:", clusters)


def update_centroids(points, clusters):
    new_m1 = np.mean(points[clusters == 1], axis=0)
    new_m2 = np.mean(points[clusters == 2], axis=0)
    print("Updated m1:", new_m1)
    print("Updated m2:", new_m2)
    return new_m1, new_m2

m1_new, m2_new = update_centroids(points, clusters)


p6_cluster = clusters[5]  
print("P6 belongs to Cluster:", p6_cluster)


population_m2 = np.sum(clusters == 2)  
print("Population of cluster around m2:", population_m2)


plt.figure(figsize=(8, 6))

plt.scatter(points[clusters == 1][:, 0], points[clusters == 1][:, 1], color='blue', label='Cluster 1')
plt.scatter(points[clusters == 2][:, 0], points[clusters == 2][:, 1], color='orange', label='Cluster 2')


plt.scatter(m1_new[0], m1_new[1], color='blue', marker='X', s=200, label='Centroid m1')
plt.scatter(m2_new[0], m2_new[1], color='orange', marker='X', s=200, label='Centroid m2')

plt.title('Final Clustering Result')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.4)
plt.ylim(0, 1)
plt.show()
