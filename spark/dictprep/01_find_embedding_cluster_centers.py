"""
Attempts to identify embedding clusters for sense disambiguation purposes and to reduce duplicate lexeme entries in
the final index/dictionary. Will return the list of cluster centers

Output format:  lexeme, sense id, cluster center for sense (base64 encoded)

General Procedure for each concept code:

For each distinct lexeme and cluster count i from 1 to spark.clr.max_wsd_clusters (inclusive)
    1) Generates i clusters on the embedding index
    2) Calculates silhouette score to approximate whether this # of clusters is appropriate

The cluster count and clusters from the maximum silhouette score is retained.
The mean point of the cluster is retained as that sense's vector representation

Requires prior completion of 00_generate_embeddings_from_nlp_artifacts.py


Required spark parameters:
    1) spark.clr.embedding_input_dir - where embeddings were written in prior step
    2) spark.clr.max_wsd_clusters - the maximal number of clusters to attempt for sense disambiguation purposes
    3) spark.clr.min_lexeme_length - filters out lexemes that are shorter than this character limit. 0=no filter
    4) spark.clr.min_wsd_freq - the minimal frequency to attempt wsd for.
    5) spark.clr.cluster_center_output_dir - Where to write output

If frequency is below min_wsd_freq, all uses are assumed to belong to the same sense
"""

import os
import base64
import struct
from typing import List

import pyspark.sql.functions as F
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Prevent KMeans on more than 1 thread due to executing multiple partitions


def encode_ndarray(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.tobytes()).decode('ascii')


def find_cluster_centers(embeddings_base64: List[str]) -> List[str]:
    # Embeddings will occasionally output a "zero'd" vector with invalid values, presumably due to an overflow(?)
    # Not sure why this happens but regardless we want to filter this out TODO investigate this
    [invalid] = struct.unpack('<d', base64.b64decode('AADA/wAAwP8='))
    embeddings: List[np.ndarray] = []

    # First, convert base64-encoded
    for embedding in embeddings_base64:
        try:
            npemb: np.ndarray = np.frombuffer(base64.b64decode(embedding))
            if np.any(np.isnan(npemb)) or not np.all(np.isfinite(npemb)):
                print("Skipping invalid embedding for NaN or not finite: ", embedding)
                continue
            if npemb.max != invalid and np.min != invalid:
                embeddings.append(npemb)
        except Exception as e:
            print("Skipping invalid embedding for other error: ", embedding)
            print(e)
    if len(embeddings) == 0:
        return []

    # Now perform WSD deconfliction if necessary, otherwise just treat as one cluster and spit out the cluster center
    if len(embeddings) >= min_wsd_freq:
        silhouettes = []
        local_centers: List[np.ndarray] = []
        # Perform k-means clustering for every k [1, min(|embeddings|, max_wsd_clusters)] and find best silhouette score
        for k in range(1, min(len(embeddings), max_wsd_clusters + 1)):
            npembeddings = np.asarray(embeddings)
            km: KMeans = KMeans(n_clusters=k, n_init=100)
            cluster_labels = km.fit_predict(npembeddings)
            silhouette_avg = silhouette_score(npembeddings, cluster_labels)
            local_centers.append(km.cluster_centers_)
            silhouettes.append(silhouette_avg)
        best_silhouette = max(silhouettes)
        packed_centers = local_centers[silhouettes.index(best_silhouette)]
        ret = []
        for center in packed_centers:
            ret.append(encode_ndarray(center))
        return ret
    else:
        # Perform single-k k-means clustering
        npembeddings = np.asarray(embeddings)
        km: KMeans = KMeans(n_clusters=1, n_init=100)
        km.fit_predict(npembeddings)
        packed_centers = km.cluster_centers_
        ret = []
        for center in packed_centers:
            ret.append(encode_ndarray(center))
        return ret


if __name__ == '__main__':
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Generate-Embeddings")
    embeddings_input_dir = spark.sparkContext.getConf().get('spark.clr.embedding_input_dir')
    max_wsd_clusters = int(spark.sparkContext.getConf().get('spark.clr.max_wsd_clusters'))
    min_wsd_freq = int(spark.sparkContext.getConf().get("spark.clr.min_wsd_freq"))
    tl_filter = int(spark.sparkContext.getConf().get("spark.clr.min_lexeme_length"))
    writedir = spark.sparkContext.getConf().get("spark.clr.cluster_center_output_dir")

    # Read in dataframe
    df: DataFrame = spark.read.format("csv").option("header", True).load(embeddings_input_dir)

    # Aggregate on lexeme to get a per-lexeme term count and a collection of relevant embeddings
    df = df.select(df[nlpio.lexeme_col_name], df[raw_embedding_col_name], F.lit(1).alias(lexeme_count_col_name))\
        .groupBy(df[nlpio.lexeme_col_name])\
        .agg(F.collect_list(df[raw_embedding_col_name]).alias(raw_embedding_col_name),
             F.sum(F.col(lexeme_count_col_name)).alias(lexeme_count_col_name))
    if tl_filter > 0:
        df = df.filter(F.length(df[lexeme_col_name]) >= tl_filter)

    # Find cluster centers
    center_search_udf = F.udf(lambda embeddings: find_cluster_centers(embeddings), ArrayType(StringType()))
    df = df.select(df[lexeme_col_name],
                   F.explode(center_search_udf(df[raw_embedding_col_name]))
                   .alias(cluster_center_col_name))
    # Add a sense ID. initialize random ordering for consistency
    df = df.select(df[nlpio.lexeme_col_name],
                   F.row_number().over(
                       Window.partitionBy(df[lexeme_col_name]).orderBy(F.rand(1))).alias(sense_id_col_name),
                   df[cluster_center_col_name])

    df.write.csv(path=writedir, mode="overwrite", header=True)


