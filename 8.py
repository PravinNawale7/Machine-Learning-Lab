import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Initialize initial medoids (first and last data points)
medoid1 = features[0]
medoid2 = features[-1]

# Define Manhattan distance function
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# K-Medoid clustering loop
while True:
    cluster1 = []
    cluster2 = []

    # Assign points to the closest medoid
    for point in features:
        dist_to_medoid1 = manhattan_distance(point, medoid1)
        dist_to_medoid2 = manhattan_distance(point, medoid2)

        if dist_to_medoid1 < dist_to_medoid2:
            cluster1.append(point)
        else:
            cluster2.append(point)

    # Recompute medoids for each cluster
    if cluster1:
        new_medoid1 = min(cluster1, key=lambda p: sum(manhattan_distance(p, other) for other in cluster1))
    else:
        new_medoid1 = medoid1

    if cluster2:
        new_medoid2 = min(cluster2, key=lambda p: sum(manhattan_distance(p, other) for other in cluster2))
    else:
        new_medoid2 = medoid2

    # If no change in medoids, the clustering is complete
    if np.array_equal(new_medoid1, medoid1) and np.array_equal(new_medoid2, medoid2):
        break

    # Update medoids
    medoid1, medoid2 = new_medoid1, new_medoid2

    # Convert clusters to numpy arrays for easy plotting
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

# Plotting the clusters and medoids
plt.scatter(cluster1[:, 0], cluster1[:, 1], color='blue', label='Cluster 1')
plt.scatter(cluster2[:, 0], cluster2[:, 1], color='green', label='Cluster 2')
plt.scatter(medoid1[0], medoid1[1], color='red', marker='X', s=200, label='Medoid 1')
plt.scatter(medoid2[0], medoid2[1], color='red', marker='X', s=200, label='Medoid 2')
plt.title('Simple K-Medoid Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Print final medoids
print("Final Medoid 1:", medoid1)
print("Final Medoid 2:", medoid2)
