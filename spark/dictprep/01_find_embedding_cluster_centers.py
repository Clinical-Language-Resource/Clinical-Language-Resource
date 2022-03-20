"""
Attempts to identify embedding clusters for sense disambiguation purposes and to reduce duplicate lexeme entries in
the final index/dictionary. Will return the list of cluster centers

Output format:  lexeme, sense id, cluster center for sense (base64 encoded)

General Procedure for each concept code:

1) Calculates gap statistic to estimate number of clusters k, up to max_wsd_clusters
2) Use KMeans to determine cluster centers.
3) If k == 1, then use the arithmetic mean of the cluster as the centerpoint

Requires prior completion of 00_generate_embeddings_from_nlp_artifacts.py


Required spark parameters:
    1) spark.clr.embedding_input_dir - where embeddings were written in prior step
    2) spark.clr.max_wsd_clusters - the maximal number of clusters to attempt for sense disambiguation purposes
    3) spark.clr.min_lexeme_length - filters out lexemes that are shorter than this character limit. 0=no filter
    4) spark.clr.min_wsd_freq - the minimal frequency to attempt wsd for.
    5) spark.clr.max_wsd_sample - the maximum number of embeddings to sample for a lexeme. 0 = no sampling/use all
    6) spark.clr.cluster_center_output_dir - Where to write output

If frequency is below min_wsd_freq, all uses are assumed to belong to the same sense
"""

import base64
import os
from typing import List

import numpy as np
import pyspark.sql.functions as F
from gap_statistic import OptimalK
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, StringType, BooleanType
from pyspark.sql.window import Window
from sklearn.cluster import KMeans

from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Prevent KMeans on more than 1 thread due to executing multiple partitions


def encode_ndarray(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.tobytes()).decode('ascii')


def embedding_valid(embedding_base64: str) -> bool:
    # Embeddings will occasionally output a "zero'd" vector with invalid values, presumably due to an overflow(?)
    # Not sure why this happens but regardless we want to filter this out TODO investigate this
    # Will also occasionally output invalid NaN and/or infinite values
    npemb: np.ndarray = np.frombuffer(base64.b64decode(embedding_base64))
    if np.any(np.isnan(npemb)) or not np.all(np.isfinite(npemb)):
        print("Skipping invalid embedding for NaN or not finite: ", embedding_base64)
        return False
    if embedding_base64.startswith('AADA/wAAwP8AAMD'):
        return False
    return True


def find_cluster_centers(embeddings_base64: List[str]) -> List[str]:
    embeddings: List[np.ndarray] = []
    # First, convert base64-encoded
    for embedding in embeddings_base64:
        try:
            npemb: np.ndarray = np.frombuffer(base64.b64decode(embedding))
            embeddings.append(npemb)
        except Exception as e:
            print("Skipping invalid embedding for other error: ", embedding)
            print(e)
    if len(embeddings) == 0:
        return []

    # Now perform WSD deconfliction if necessary, otherwise just treat as one cluster and spit out the cluster center
    if len(embeddings) >= min_wsd_freq:
        npembeddings = np.asarray(embeddings)
        # Use the gap statistic to estimate k for K-Means
        optimal_k = OptimalK()
        n_clusters = optimal_k(npembeddings, cluster_array=range(1, max_wsd_clusters + 1))
        if n_clusters > 1:
            km: KMeans = KMeans(n_clusters=n_clusters, n_init=100)
            km.fit_predict(npembeddings)
            ret: List[str] = []
            for center in km.cluster_centers_:
                ret.append(encode_ndarray(center))
            return ret
        else:
            return [encode_ndarray(np.asarray(embeddings).mean(axis=0, dtype=np.float64))]
    else:
        # Just return the arithmetic mean
        return [encode_ndarray(np.asarray(embeddings).mean(axis=0, dtype=np.float64))]


if __name__ == '__main__':
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Generate-Embeddings")
    embeddings_input_dir = spark.sparkContext.getConf().get('spark.clr.embedding_input_dir')
    max_wsd_clusters = int(spark.sparkContext.getConf().get('spark.clr.max_wsd_clusters'))
    min_wsd_freq = int(spark.sparkContext.getConf().get("spark.clr.min_wsd_freq"))
    tl_filter = int(spark.sparkContext.getConf().get("spark.clr.min_lexeme_length"))
    max_wsd_sample = int(spark.sparkContext.getConf().get("spark.clr.max_wsd_sample"))
    writedir = spark.sparkContext.getConf().get("spark.clr.cluster_center_output_dir")

    # Read in dataframe
    df: DataFrame = spark.read.format("csv").option("header", True).load(embeddings_input_dir)

    # Filter invalid embeddings
    embedding_valid_udf = F.udf(lambda s: embedding_valid(s), BooleanType())
    df = df.filter(embedding_valid_udf(df[raw_embedding_col_name]))

    # Sample if necessary
    if max_wsd_sample > 0:
        df = df.withColumn(lexeme_sample_idx_col_name,
                           F.row_number().over(Window.partitionBy(df[lexeme_col_name]).orderBy(F.rand())))
        df = df.filter(df[lexeme_sample_idx_col_name] <= max_wsd_sample)

    # Aggregate on lexeme to get a per-lexeme term count and a collection of relevant embeddings
    df = df.select(df[lexeme_col_name], df[raw_embedding_col_name], F.lit(1).alias(lexeme_count_col_name))\
        .groupBy(df[lexeme_col_name])\
        .agg(F.collect_list(df[raw_embedding_col_name]).alias(raw_embedding_col_name),
             F.sum(F.col(lexeme_count_col_name)).alias(lexeme_count_col_name))
    if tl_filter > 0:
        df = df.filter(F.length(df[lexeme_col_name]) >= tl_filter)

    # Find cluster centers
    center_search_udf = F.udf(lambda embeddings: find_cluster_centers(embeddings), ArrayType(StringType()))
    df = df.select(df[lexeme_col_name],
                   F.explode(center_search_udf(df[raw_embedding_col_name]))
                   .alias(cluster_center_col_name),
                   df[lexeme_count_col_name])
    # Add a sense ID. initialize random ordering for consistency
    df = df.select(df[nlpio.lexeme_col_name],
                   F.row_number().over(
                       Window.partitionBy(df[lexeme_col_name]).orderBy(F.rand(1))).alias(sense_id_col_name),
                   df[cluster_center_col_name],
                   df[lexeme_count_col_name])

    df.write.csv(path=writedir, mode="overwrite", header=True)


