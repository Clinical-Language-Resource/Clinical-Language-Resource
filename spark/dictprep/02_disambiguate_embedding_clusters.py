"""
Attempts to identify embedding clusters for sense disambiguation purposes and to reduce duplicate lexeme entries in
the final index/dictionary. Will return the list of cluster centers

Output format:  lexeme, sense id, cluster center for sense (base64 encoded)

General Procedure for each concept code:

1) Uses concept normalization to naively determine number of senses. Note that this approach is imperfect and
   alternate strategies (such as using the gap statistic and/or silhouette scores) to find an optimal K for clustering
2) Use KMeans to construct k number of clusters, where k = number of senses.
3) If k == 1, then use the arithmetic mean of the cluster as the centerpoint

Requires prior completion of 01_generate_embeddings_from_nlp_artifacts.py


Required spark parameters:
    1) spark.clr.embedding_input_dir - where embeddings were written in prior step
    3) spark.clr.min_lexeme_length - filters out lexemes that are shorter than this character limit. 0=no filter
    4) spark.clr.min_wsd_freq - the minimal frequency to attempt wsd for.
    5) spark.clr.max_wsd_sample - the maximum number of embeddings to sample for a lexeme. 0 = no sampling/use all
    6) spark.clr.cluster_center_output_dir - Where to write output
    7) spark.clr.min_cluster_size - the absolute minimum cluster size, otherwise filter out
    8) spark.clr.min_cluster_size_prop - Minimum cluster size as a proportion of sample size (up to max_wsd_sample)

If frequency is below min_wsd_freq, all uses are assumed to belong to the same sense. If both min_cluster_size and
min_cluster_size_prop is defined, whichever is the larger of the two will be used.
"""

import base64
import math
import os
from typing import List, Tuple

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, ArrayType, StructType, DataType, StructField, IntegerType
from pyspark.sql.window import Window
from sklearn.cluster import KMeans

from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

os.environ['OPENBLAS_NUM_THREADS'] = "1"  # Prevent KMeans on more than 1 thread due to executing multiple partitions


def encode_ndarray(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.tobytes()).decode('ascii')


def num_samples_in_cluster(cluster_idx: int, labels) -> int:
    return len(np.where(labels == cluster_idx)[0])


def find_cluster_centers(embeddings_base64: List[str], num_clusters: int) -> List[Tuple[int, str]]:
    embeddings: List[np.ndarray] = []
    # First, convert base64-encoded
    for embedding in embeddings_base64:
        npemb: np.ndarray = np.frombuffer(base64.b64decode(embedding), dtype="float32")
        embeddings.append(npemb)
    if len(embeddings) == 0:
        return []

    cluster_size_limit = max(math.floor(min_cluster_size_prop * len(embeddings)), min_cluster_size)

    # Now perform WSD deconfliction if necessary, otherwise just treat as one cluster and spit out the cluster center
    if len(embeddings) >= min_wsd_freq and len(embeddings) >= cluster_size_limit:
        npembeddings = np.asarray(embeddings)
        if num_clusters > 1:
            km: KMeans = KMeans(n_clusters=num_clusters, n_init=100)
            km.fit_predict(npembeddings)
            ret: List[Tuple[int, str]] = []
            idx: int = 0
            for center in km.cluster_centers_:
                cluster_size = num_samples_in_cluster(idx, km.labels_)
                idx += 1
                if cluster_size >= cluster_size_limit:
                    ret.append((cluster_size, encode_ndarray(center)))
            return ret
        else:
            return [(len(embeddings), encode_ndarray(np.asarray(embeddings).mean(axis=0, dtype=np.float64)))]
    else:
        if len(embeddings) >= cluster_size_limit:
            # Just return the arithmetic mean
            return [(len(embeddings), encode_ndarray(np.asarray(embeddings).mean(axis=0, dtype=np.float64)))]
        else:
            return []


def find_cluster_centers_schema() -> DataType:
    return ArrayType(StructType([
        StructField(cluster_size_col_name, IntegerType(), False),
        StructField(cluster_center_col_name, StringType(), False)
    ]))


if __name__ == "__main__":
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Find-Embedding-Centers")
    embeddings_input_dir = spark.sparkContext.getConf().get('spark.clr.embedding_input_dir')
    min_wsd_freq = int(spark.sparkContext.getConf().get("spark.clr.min_wsd_freq"))
    tl_filter = int(spark.sparkContext.getConf().get("spark.clr.min_lexeme_length"))
    max_wsd_sample = int(spark.sparkContext.getConf().get("spark.clr.max_wsd_sample"))
    writedir = spark.sparkContext.getConf().get("spark.clr.cluster_center_output_dir")
    min_cluster_size = int(spark.sparkContext.getConf().get("spark.clr.min_cluster_size"))
    min_cluster_size_prop = float(spark.sparkContext.getConf().get("spark.clr.min_cluster_size_prop"))

    # Estimate sense counts from raw data by normalizing lexemes and grouping by some semantic grouping
    normalize_lexeme_udf = F.udf(lambda lex, concept: nlpio.normalize_lexeme_concept(lex, concept), StringType())
    normalize_concept_udf = F.udf(lambda lex, concept: nlpio.normalize_concept_grouping(lex, concept), StringType())
    sense_counts: DataFrame = nlpio.get_nlp_artifact_table(spark)
    sense_counts = sense_counts.select(
        normalize_lexeme_udf(sense_counts[lexical_variant_col_name], sense_counts[concept_id_col_name]).alias(
            lexeme_col_name),
        normalize_concept_udf(
            sense_counts[lexical_variant_col_name],
            sense_counts[concept_id_col_name]
        ).alias(concept_code_col_name)
    )
    sense_counts = sense_counts.groupBy(
        lexeme_col_name
    ).agg(
        F.size(F.collect_set(sense_counts[concept_code_col_name])).alias(sense_count_col_name)
    ).select(
        lexeme_col_name,
        sense_count_col_name
    )

    # Load embeddings
    df: DataFrame = spark.read.format("csv").option("header", True).load(embeddings_input_dir)

    # Cut down to distinct embeddings to prevent duplicate sentences across different notes (e.g. templates)
    df = df.select(df[concept_code_col_name],
                   df[lexeme_col_name],
                   df[raw_embedding_col_name]).distinct()

    # Sample if necessary
    if max_wsd_sample > 0:
        df = df.withColumn(lexeme_sample_idx_col_name,
                           F.row_number().over(Window.partitionBy(df[lexeme_col_name]).orderBy(F.rand())))
        df = df.filter(df[lexeme_sample_idx_col_name] <= max_wsd_sample)

    # Aggregate on lexeme to get a per-lexeme term count and a collection of relevant embeddings
    df = df.select(df[lexeme_col_name], df[raw_embedding_col_name], F.lit(1).alias(lexeme_count_col_name)) \
        .groupBy(df[lexeme_col_name]) \
        .agg(F.collect_list(df[raw_embedding_col_name]).alias(raw_embedding_col_name),
             F.sum(F.col(lexeme_count_col_name)).alias(lexeme_count_col_name))
    if tl_filter > 0:
        df = df.filter(F.length(df[lexeme_col_name]) >= tl_filter)

    # Join with sense_counts to get number of clusters to use
    df = df.join(sense_counts, df[lexeme_col_name] == sense_counts[lexeme_col_name]).select(
        df[lexeme_col_name],
        df[raw_embedding_col_name],
        df[lexeme_count_col_name],
        sense_counts[sense_count_col_name]
    )

    # Now find cluster centers
    center_search_udf = F.udf(
        lambda embeddings, cluster_cnt: find_cluster_centers(embeddings, cluster_cnt),
        find_cluster_centers_schema()
    )
    df = df.select(
        df[lexeme_col_name],
        F.explode(
            center_search_udf(
                df[raw_embedding_col_name],
                df[sense_count_col_name]
            )
        ).alias(cluster_info_struct_name),
        df[lexeme_count_col_name]
    )

    # And unpack the struct
    df = df.select(df[lexeme_col_name],
                   F.col(cluster_info_struct_name + "." + cluster_size_col_name).alias(cluster_size_col_name),
                   F.col(cluster_info_struct_name + "." + cluster_center_col_name).alias(cluster_center_col_name),
                   df[lexeme_count_col_name])

    # Add a random sense ID
    df = df.select(df[nlpio.lexeme_col_name],
                   F.row_number().over(
                       Window.partitionBy(df[lexeme_col_name]).orderBy(F.rand())).alias(sense_id_col_name),
                   df[cluster_center_col_name],
                   df[cluster_size_col_name],
                   df[lexeme_count_col_name])

    df.write.csv(path=writedir, mode="overwrite", header=True)
