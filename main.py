import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min, euclidean_distances
from sklearn.preprocessing import MinMaxScaler

# Read data
df = pd.read_csv('data-example2223.csv')
df = df.dropna()
X_train = df.to_numpy()

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

initial_k = 100
kmeans_initial = KMeans(n_clusters=initial_k, random_state=42)
df['cluster'] = kmeans_initial.fit_predict(X_train)

plt.figure(figsize=(10, 8))

for cluster_id_initial in range(initial_k):
    cluster_data_initial = X_train[df['cluster'] == cluster_id_initial]
    plt.scatter(cluster_data_initial[:, 0], cluster_data_initial[:, 1], label=f'Cluster {cluster_id_initial}', alpha=0.5)

plt.title('Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

connected_clusters = list(range(initial_k))
pairwise_distances_centers = squareform(pdist(kmeans_initial.cluster_centers_))

while len(connected_clusters) > 5:
    # Find the closest clusters among the remaining connected clusters
    min_distance = np.inf
    merge_clusters = (0, 0)

    for i in connected_clusters:
        for j in connected_clusters:
            if i != j and pairwise_distances_centers[i, j] < min_distance:
                min_distance = pairwise_distances_centers[i, j]
                merge_clusters = (i, j)

    # Merge the two closest clusters
    connected_clusters.remove(merge_clusters[1])

    # Calculate the new cluster center for the merged cluster
    merged_points = X_train[df['cluster'].isin(merge_clusters)]
    new_cluster_center = np.mean(merged_points, axis=0)

    # Update the 'cluster' column with the merged cluster
    df['cluster'] = np.where(df['cluster'] == merge_clusters[1], merge_clusters[0], df['cluster'])

    # Update the pairwise distances after merging
    for i in connected_clusters:
        if i != merge_clusters[0]:
            # Calculate the new pairwise distance between the merged cluster and other clusters
            pairwise_distances_centers[merge_clusters[0], i] = np.linalg.norm(new_cluster_center - np.mean(X_train[df['cluster'] == i], axis=0))
            pairwise_distances_centers[i, merge_clusters[0]] = np.linalg.norm(new_cluster_center - np.mean(X_train[df['cluster'] == i], axis=0))


plt.figure(figsize=(10, 8))

preferable_colors = ['red', 'blue', 'green', 'purple', "orange", "magenta", "cyan"]

for cluster_id_initial in range(initial_k):
    cluster_data_initial = X_train[df['cluster'] == cluster_id_initial]
    plt.scatter(
        cluster_data_initial[:, 0],
        cluster_data_initial[:, 1],
        label=f'Cluster {cluster_id_initial}',
        alpha=0.5,
        c=preferable_colors[cluster_id_initial % len(preferable_colors)]
    )

plt.title('Scatter Plot of 5 Final Clusters with Preferable Colors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
