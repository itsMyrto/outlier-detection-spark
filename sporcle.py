import sys
from time import time
import numpy as np
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col
from scipy.spatial.distance import pdist, squareform
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans


def min_max_scaling(column):
    min_value = df.agg(F.min(column)).collect()[0][0]
    max_value = df.agg(F.max(column)).collect()[0][0]
    scaled_column = (col(column) - min_value) / (max_value - min_value)
    return scaled_column.alias(f"{column}_scaled")


if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """

    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    # Extract the filename from the command line argument
    filename = sys.argv[1]

    start = time()

    spark = SparkSession.builder \
        .appName("example") \
        .config("spark.driver.port", "7071") \
        .getOrCreate()

    # Reading the data from the CSV and dropping null rows
    df = spark.read.csv(filename, header=False, inferSchema=True).na.drop().withColumnRenamed("_c0",
                                                                                                            "x").withColumnRenamed(
        "_c1", "y")
    df = df.withColumn("x_scaled", min_max_scaling("x")).withColumn("y_scaled", min_max_scaling("y"))

    assembler = VectorAssembler(inputCols=["x_scaled", "y_scaled"], outputCol="features")
    df = assembler.transform(df)

    k = 60
    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df)
    df = model.transform(df)

    df = df.select("cluster", "x_scaled", "y_scaled", "x", "y")
    simple_df = df.select("cluster")

    cluster_centers_np = model.clusterCenters()

    cluster_center_lists = [[center[0], center[1]] for center in cluster_centers_np]
    cluster_centers_lists = np.array(cluster_center_lists)

    # This list at first contains all the cluster ids, but eventually it will end with 5 clusters
    final_clusters = list(range(k))

    # Calculating the distance between all the cluster centers. This will be a kxk matrix.
    # Matrix[i, j] contains the Euclidean distance between the centers of cluster i and j
    pairwise_distances_centers = squareform(pdist(cluster_center_lists))

    # Convert the PySpark DataFrame to a Pandas DataFrame
    cluster_df = df.select("cluster", "x_scaled", "y_scaled").toPandas().to_numpy()

    new_column = cluster_df[:, 0].copy()

    dictionary = {}

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

        # Find rows where the first column is equal to i or j
        selected_rows = cluster_df[(cluster_df[:, 0] == merge_clusters[0]) | (cluster_df[:, 0] == merge_clusters[1])]

        # Calculate mean of the second and third columns separately
        mean_values = np.mean(selected_rows[:, 1:], axis=0)

        # Update the first column where it is equal to j to i
        cluster_df[cluster_df[:, 0] == merge_clusters[1], 0] = merge_clusters[0]

        # Updating the pairwise distances after merging
        for i in final_clusters:
            if i != merge_clusters[0]:
                # Calculating the new pairwise distance between the merged cluster and other clusters
                pairwise_distances_centers[merge_clusters[0], i] = np.linalg.norm(
                    mean_values - np.mean(cluster_df[cluster_df[:, 0] == i, 1:], axis=0))
                pairwise_distances_centers[i, merge_clusters[0]] = np.linalg.norm(
                    mean_values - np.mean(cluster_df[cluster_df[:, 0] == i, 1:], axis=0))

    another_column = cluster_df[:, 0]

    for i in range(0, len(new_column)):
        dictionary[new_column[i]] = int(another_column[i])

    df = df.withColumn("cluster", F.udf(lambda x: dictionary[x])("cluster"))

    # Define a Window specification based on the 'cluster' column
    window_spec = Window.partitionBy("cluster")

    # Calculate mean and standard deviation for 'x_scaled' and 'y_scaled' within each cluster
    df = df.withColumn("mean_x", F.avg("x_scaled").over(window_spec)) \
        .withColumn("stddev_x", F.stddev("x_scaled").over(window_spec)) \
        .withColumn("mean_y", F.avg("y_scaled").over(window_spec)) \
        .withColumn("stddev_y", F.stddev("y_scaled").over(window_spec))

    # Calculate z-scores for 'x_scaled' and 'y_scaled'
    df = df.withColumn("z_score_x", (F.col("x_scaled") - F.col("mean_x")) / F.col("stddev_x")) \
        .withColumn("z_score_y", (F.col("y_scaled") - F.col("mean_y")) / F.col("stddev_y"))

    # Calculate absolute z-scores
    df = df.withColumn("abs_z_score_x", F.abs(F.col("z_score_x"))) \
        .withColumn("abs_z_score_y", F.abs(F.col("z_score_y")))

    df = df.withColumn("aspect_ratio",
                       (F.max("x_scaled").over(window_spec).cast("double") - F.min("x_scaled").over(window_spec).cast(
                           "double")) /
                       (F.max("y_scaled").over(window_spec).cast("double") - F.min("y_scaled").over(window_spec).cast(
                           "double")))

    # Set threshold based on aspect ratio
    df = df.withColumn("threshold", F.when(F.col("aspect_ratio") > 1.7, 3).otherwise(2.28))

    # Identify outliers
    df = df.withColumn("is_outlier_x", F.when(F.col("abs_z_score_x") > F.col("threshold"), 1).otherwise(0)) \
        .withColumn("is_outlier_y", F.when(F.col("abs_z_score_y") > F.col("threshold"), 1).otherwise(0))

    outliers_df = df.filter((F.col("is_outlier_x") == 1) | (F.col("is_outlier_y") == 1)).select("x", "y")
    outliers_df.show()

    print("Runtime: ", time() - start, "seconds")

    spark.stop()
