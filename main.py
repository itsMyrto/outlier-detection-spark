import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from keras import Sequential
# from keras.src.engine.input_layer import InputLayer
# from keras.src.layers import Dense
# from keras.src.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from time import time

start = time()

# Reading the data from the csv and storing it to a panda dataframe
df = pd.read_csv('data-example2223.csv')

# Dropping all the rows that contain null
df = df.dropna()

# Transforming the pandas dataframe to a numpy array in order to run the machine learning algorithms
X_train = df.to_numpy()

# Adding better precision in the pandas dataframe to save the normalized values
pd.set_option('display.float_format', '{:.8f}'.format)

# Normalize data using the MinMax between [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Saving the transformed data back to the df
df['x'] = X_train[:, 0]
df['y'] = X_train[:, 1]

# Running the kmeans algorithm using k>>5
k = 100
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

# Applying the kmeans algorithm and then save for every point the cluster id it belongs in a new column in the dataframe
df['cluster'] = kmeans.fit_predict(X_train)

# Plotting the initial clusters with k>>5
for cluster_id in range(k):
    cluster_data = X_train[df['cluster'] == cluster_id]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_id}', alpha=0.5)

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


# APPLYING DBSCAN TO EACH CLUSTER TO DETECT THE OUTLIERS

# Best parameters for DBSCAN
min_samples = 22
eps = 0.035

# Variables for plotting the clusters with their outliers without the normalization
count = 0
cluster_outlier_indexes = []

# This loop iterates the data of each cluster based on the labels that were assigned to the points by the previous process
for cluster_id in np.unique(df['cluster']):
    # Get separately the normalized points of each cluster that was computed and saved earlier in the df
    cluster = df[df['cluster'] == cluster_id][['x', 'y']]
    # Transforming the pandas dataframe to a numpy array in order to run DBSCAN for outlier detection
    cluster = np.array(cluster)

    # start k-plots
    # neighbors = NearestNeighbors(n_neighbors=min_samples)
    # neighbors_fit = neighbors.fit(cluster)
    # distances, indices = neighbors_fit.kneighbors(cluster)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()
    # eps = [0.004, 0.004, 0.005, 0.005, 0.005] -> result for min_samples=80
    # end k-plots

    # start DBSCAN
    # Running the DBSCAN algorithm using the best parameters we found
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dbscan.fit(cluster)
    labels = dbscan.labels_
    # Saving the indexes of the outliers in each cluster using the labels from DBSCAN
    outlier_indexes = np.where(labels == -1)[0]
    # end DBSCAN

    # start Z-Scores
    # z_scores = (cluster - cluster.mean(axis=0)) / cluster.std(axis=0)
    # # Calculate absolute value of Z-Score for each point
    # abs_z_scores = np.max(np.abs(z_scores), axis=1)
    # # Set a threshold for outliers (e.g., Z-Score greater than 3)
    # threshold = 2.288
    # outliers = np.where(abs_z_scores > threshold)[0]
    # end Z-Scores

    # Print the outliers of each cluster if they exist
    print("Outliers of Cluster ", count, ": ")
    if len(outlier_indexes) == 0:
        # If a cluster has no outliers then print "No outliers found"
        print("No outliers found")
    else:
        # Get the normalized points of the outliers for the cluster data
        outliers = cluster[outlier_indexes]
        # Inverse Transforming the points of the outliers to print the original data and not the normalized
        original_outliers = scaler.inverse_transform(outliers)
        print(original_outliers)

    # keep the outlier_indexes so that we can plot them later
    cluster_outlier_indexes.append(outlier_indexes)
    count += 1


# Plotting the final clusters and their outliers
count = 0
for cluster_id in np.unique(df['cluster']):
    # Get separately the normalized points of each cluster that was computed and saved earlier in the df
    cluster = df[df['cluster'] == cluster_id][['x', 'y']]
    # Transforming the pandas dataframe to a numpy array in order to run DBSCAN for outlier detection
    cluster = np.array(cluster)
    # Inverse Transforming the points of the cluster to plot the original data and not the normalized
    original_cluster = scaler.inverse_transform(cluster)
    # Get the indexes of the cluster's outliers
    outliers = cluster_outlier_indexes[count]

    # Plot the original points of each cluster
    plt.scatter(
        original_cluster[:, 0],
        original_cluster[:, 1],
        label=f'Cluster {cluster_id}',
        alpha=0.5,
        c=preferable_colors[count]
    )
    # Plot on top of the original points of each cluster the outliers that were found
    plt.scatter(original_cluster[outliers, 0], original_cluster[outliers, 1], c="red", marker="x")
    count += 1

# Show final plot that contains every cluster and it's outliers
plt.title('Scatter Plot of 5 Final Clusters and their Outliers')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


print("Runtime: ", time()-start)


# ΝΕΥΡΩΝΙΚΟ ΔΙΚΤΥΟ

#
# outliers = []
#
# for cluster_id_final in range(5):
#
#     cluster_data_final = X_train[df['cluster'] == final_clusters[cluster_id_final]]
#
#     # Train autoencoder
#     OPTIMIZER_1 = Adam(learning_rate=0.001)
#     autoencoder = Sequential()
#     autoencoder.add(InputLayer(input_shape=(2,)))
#     autoencoder.add(Dense(units=2, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=3, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=2, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=1, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=2, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=3, activation="linear", use_bias=True))
#     autoencoder.add(Dense(units=2, activation="linear", use_bias=True))
#     autoencoder.compile(optimizer=OPTIMIZER_1, loss='mean_squared_error')
#     autoencoder.fit(cluster_data_final, cluster_data_final, epochs=25, batch_size=32, shuffle=True)
#
#     # Predict and calculate reconstruction errors
#     reconstructed_data = autoencoder.predict(cluster_data_final)
#     reconstruction_errors = np.mean(np.square(cluster_data_final - reconstructed_data), axis=1)
#
#     # Set a threshold for outliers within the cluster
#     threshold = np.percentile(reconstruction_errors, 99.9)
#
#     # Identify outliers within the cluster
#     cluster_outliers = cluster_data_final[reconstruction_errors > threshold]
#     outliers.append(cluster_outliers)
#
# # Concatenate outliers from all clusters
# all_outliers = np.concatenate(outliers, axis=0)
#
#
# # Plot original data and connected clusters
# for cluster_id_final in range(5):
#     cluster_data_final = df[df['cluster'] == final_clusters[cluster_id_final]]
#     plt.scatter(
#         cluster_data_final['x'],
#         cluster_data_final['y'],
#         label=f'Final Cluster {cluster_id_final}',
#         alpha=0.5,
#         c=preferable_colors[cluster_id_final % len(preferable_colors)]
#     )
#
# all_outliers = np.concatenate(outliers)
#
# # Plot outliers
# plt.scatter(
#     all_outliers[:, 0],
#     all_outliers[:, 1],
#     label='Outliers',
#     color='black',
#     marker='x'
# )
#
# plt.title('Outlier Detection with Autoencoders')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
