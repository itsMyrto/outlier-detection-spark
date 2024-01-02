import math
import sys
from imp import reload

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from scipy.stats import median_abs_deviation
import numpy as np
from pyspark.sql import functions as F, Window
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, when, avg, mean, stddev, monotonically_increasing_id, row_number, lit, udf
from scipy.spatial.distance import pdist, squareform, euclidean
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
import dbscan





def min_max_scaling(column):
    min_value = df.agg(F.min(column)).collect()[0][0]
    max_value = df.agg(F.max(column)).collect()[0][0]
    scaled_column = (col(column) - min_value) / (max_value - min_value)
    return scaled_column.alias(f"{column}_scaled")


if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """

    spark = SparkSession.builder \
        .appName("example") \
        .config("spark.driver.port", "7071") \
        .config("spark.driver.bindAddress", "192.168.1.9") \
        .getOrCreate()

    # Reading the data from the CSV and dropping null rows
    df = spark.read.csv('data-example2223.csv', header=False, inferSchema=True).na.drop().withColumnRenamed("_c0",
                                                                                                            "x").withColumnRenamed(
        "_c1", "y")
    df = df.withColumn("x_scaled", min_max_scaling("x")).withColumn("y_scaled", min_max_scaling("y"))

    assembler = VectorAssembler(inputCols=["x_scaled", "y_scaled"], outputCol="features")
    df = assembler.transform(df)

    k = 60
    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df)
    df = model.transform(df)

    df.show(5, truncate=False)

    df = df.select("cluster", "x_scaled", "y_scaled")
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
    cluster_df = df.toPandas().to_numpy()

    new_column = cluster_df[:, 0]

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


    print(cluster_df)

    another_column = cluster_df[:, 0]
    print(another_column)

    for i in range(0, len(new_column)):
        dictionary[new_column[i]] = another_column[i]

    print(len(dictionary))
    print(dictionary)

    print("here")

    df_ = df.withColumn("cluster", F.udf(lambda x: dict[x], IntegerType())("cluster"))

    df_.show()

    preferable_colors = ["#0766AD", "#29ADB2", "#D2DE32", "#B15EFF", "#952323"]
    count = 0

    # Plotting the initial clusters
    for cluster_id in final_clusters:
        cluster_points = cluster_df[cluster_df[:, 0] == cluster_id]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 2],
                    label=f'Cluster {cluster_id}', alpha=0.5, c=preferable_colors[count])
        count += 1

    # Display the scatter plot
    plt.title('Scatter Plot')
    plt.xlabel('x scaled')
    plt.ylabel('y scaled')
    plt.show()


    # Assuming 'final_cluster' is the column you want to group by
    window_spec = Window.partitionBy("cluster")

    # Calculate median and mad for 'x_scaled' and 'y_scaled'
    df_stats = df.groupBy("cluster").agg(
        F.expr("percentile_approx(x_scaled, 0.5)").alias("median_x"),
        F.expr("percentile_approx(y_scaled, 0.5)").alias("median_y"),
        F.expr("percentile_approx(abs(x_scaled), 0.5)").alias("mad_x"),
        F.expr("percentile_approx(abs(y_scaled), 0.5)").alias("mad_y")
    )

    # Join the aggregated statistics with the main DataFrame
    df = df.join(df_stats, "cluster", "left_outer")

    # Calculate modified z-scores for 'x_scaled' and 'y_scaled'
    df = df.withColumn("z_score_x",
                       0.6745 * F.when(df["mad_x"] != 0, (df["x_scaled"] - df["median_x"]) / df["mad_x"]).otherwise(0))
    df = df.withColumn("z_score_y",
                       0.6745 * F.when(df["mad_y"] != 0, (df["y_scaled"] - df["median_y"]) / df["mad_y"]).otherwise(0))

    df = df.withColumn("aspect_ratio",
                       (F.max("x_scaled").over(window_spec).cast("double") - F.min("x_scaled").over(window_spec).cast(
                           "double")) /
                       (F.max("y_scaled").over(window_spec).cast("double") - F.min("y_scaled").over(window_spec).cast(
                           "double")))

    # Calculate threshold multipliers dynamically based on shape
    df = df.withColumn("threshold_multiplier_x", F.when(df["aspect_ratio"] > 1, 1.0).otherwise(2.0))
    df = df.withColumn("threshold_multiplier_y", F.when(df["aspect_ratio"] > 1, 1.0).otherwise(2.0))

    # Set dynamic thresholds based on MAD and shape
    df = df.withColumn("threshold_x", F.col("threshold_multiplier_x") * F.col("mad_x"))
    df = df.withColumn("threshold_y", F.col("threshold_multiplier_y") * F.col("mad_y"))

    # Identify outliers
    df = df.withColumn("outlier_x", F.abs(df["z_score_x"]) > df["threshold_x"])
    df = df.withColumn("outlier_y", F.abs(df["z_score_y"]) > df["threshold_y"])
    df = df.withColumn("outlier", (df["outlier_x"] | df["outlier_y"]).cast("int"))


    df.printSchema()

    df_outliers = df.toPandas().to_numpy()

    print(df_outliers.count())


    # Stop the Spark session
    spark.stop()
