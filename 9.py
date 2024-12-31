import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the dataset (replace 'shopping_trends.csv' with your actual file)
df = pd.read_csv('shopping_trends.csv')

# Data preprocessing
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean

# Standardize the data
X_scaled = StandardScaler().fit_transform(df)

# Generate the linkage matrix and plot dendrogram
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply Agglomerative Clustering
n_clusters = 3
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['Cluster'] = hc.fit_predict(X_scaled)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Scatter plot for clusters based on original features
plt.figure(figsize=(8, 5))
plt.scatter(df.loc[df['Cluster'] == 0].iloc[:, 0], df.loc[df['Cluster'] == 0].iloc[:, 1], s=50, c='red', label='Cluster 0')
plt.scatter(df.loc[df['Cluster'] == 1].iloc[:, 0], df.loc[df['Cluster'] == 1].iloc[:, 1], s=50, c='blue', label='Cluster 1')
plt.scatter(df.loc[df['Cluster'] == 2].iloc[:, 0], df.loc[df['Cluster'] == 2].iloc[:, 1], s=50, c='green', label='Cluster 2')

plt.title('Clusters of Shopping Trends Data')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend()
plt.grid()
plt.show()

# Print cluster counts
print("Cluster counts:")
print(df['Cluster'].value_counts())
