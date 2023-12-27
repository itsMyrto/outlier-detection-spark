import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.src.engine.input_layer import InputLayer
from keras.src.optimizers import SGD, Adam
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


