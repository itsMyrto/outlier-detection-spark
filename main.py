import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Reading the data from the csv and storing it to a panda dataframe
df = pd.read_csv('data-example2223.csv')

# Dropping all the rows that contain null
df = df.dropna()

# Transforming the pandas dataframe to a numpy array in order to run the machine learning algorithms
X_train = df.to_numpy()

# Normalize data using the MinMax between [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Running the kmeans algorithm using k>>5
k = 300
kmeans = KMeans(n_clusters=k, random_state=42)

# Applying the kmeans algorithm and then save for every point the cluster id it belongs in a new column in the dataframe
df['cluster'] = kmeans.fit_predict(X_train)

# Plotting the initial clusters with k>>5
for cluster_id_initial in range(k):
    cluster_data_initial = X_train[df['cluster'] == cluster_id_initial]
    plt.scatter(cluster_data_initial[:, 0], cluster_data_initial[:, 1], label=f'Cluster {cluster_id_initial}', alpha=0.5)

plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# This list at first contains all the cluster ids, but eventually it will end with 5 clusters
final_clusters = list(range(k))

# Calculating the distance between all the cluster centers. This will be a kxk matrix.
# Matrix[i, j] contains the Euclidean distance between the centers of cluster i and j
pairwise_distances_centers = squareform(pdist(kmeans.cluster_centers_))

# This loop merges clusters based on the center distances, until 5 clusters are left

while len(final_clusters) > 5:
    # Find the closest clusters among the remaining connected clusters
    # Setting the min distance equal to infinity and the merging clusters as 0,0
    min_distance = np.inf
    merge_clusters = (0, 0)

    # Iterating through all the clusters in order to find the minimum distance between two clusters
    for i in final_clusters:
        for j in final_clusters:
            if i != j and pairwise_distances_centers[i, j] < min_distance:
                min_distance = pairwise_distances_centers[i, j]
                merge_clusters = (i, j)

    # Removing the cluster j from the list
    final_clusters.remove(merge_clusters[1])

    # Calculating the new cluster center for the merged cluster i.e. combining the points from cluster i and j in order to find the new center
    merged_points = X_train[df['cluster'].isin(merge_clusters)]
    new_cluster_center = np.mean(merged_points, axis=0)

    # Updating the 'cluster' column with the merged cluster. Points with cluster == j, now have cluster == i
    df['cluster'] = np.where(df['cluster'] == merge_clusters[1], merge_clusters[0], df['cluster'])

    # Updating the pairwise distances after merging
    for i in final_clusters:
        if i != merge_clusters[0]:
            # Calculating the new pairwise distance between the merged cluster and other clusters
            pairwise_distances_centers[merge_clusters[0], i] = np.linalg.norm(new_cluster_center - np.mean(X_train[df['cluster'] == i], axis=0))
            pairwise_distances_centers[i, merge_clusters[0]] = np.linalg.norm(new_cluster_center - np.mean(X_train[df['cluster'] == i], axis=0))


# Plotting the final clusters
preferable_colors = ["#0766AD", "#29ADB2", "#D2DE32", "#B15EFF", "#952323"]
count = 0

for cluster in final_clusters:
    cluster_data = X_train[df['cluster'] == cluster]
    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f'Cluster {cluster}',
        alpha=0.5,
        c=preferable_colors[count]
    )
    count += 1

plt.title('Scatter Plot of 5 Final Clusters')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
