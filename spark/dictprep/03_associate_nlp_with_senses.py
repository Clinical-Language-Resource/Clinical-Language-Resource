"""
Optional: Associates NLP outputs with cluster sense ids at a per-document level. Useful later for WSD-related tasks and
sense definitions in the dictionary, but is not critical to core functionality

Association is done via cosine similarity to the individual vector to the various cluster centers of corresponding
(lexeme, sense) pairs

Requires completion of 02_find_embedding_cluster_centers.py

Output format: note id, concept id, lexeme, sense id

Required spark parameters:
    1) spark.clr.embedding_input_dir - where embeddings were written in prior step
    2) spark.clr.cluster_center_input_dir - cluster center output from 01_find_embedding_cluster_centers
    3) spark.clr.sense_associations_output_dir - Where to write results
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import BooleanType, FloatType

from clinicallanguageresource.dictprep.site_modify import sparkutils
from clinicallanguageresource.dictprep.site_modify.column_names import *

import numpy as np
import base64
import pyspark.sql.functions as F


def embedding_valid(embedding_base64: str) -> bool:
    # Embeddings will occasionally output a NaN vector with invalid values, presumably due to an overflow(?)
    # Not sure why this happens but regardless we want to filter this out TODO investigate this
    npemb: np.ndarray = np.frombuffer(base64.b64decode(embedding_base64), dtype="float32")
    if np.any(np.isnan(npemb)) or not np.all(np.isfinite(npemb)):
        return False
    return True


def cos_sim(embedding1_base64: str, embedding2_base64: str) -> float:
    """
    :return: The cosine similarity between two vectors
    """
    emb1 = np.frombuffer(base64.b64decode(embedding1_base64), dtype="float32")
    emb2 = np.frombuffer(base64.b64decode(embedding2_base64), dtype="float32")
    return np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))


if __name__ == "__main__":
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Find-Disambiguation-Contexts")
    embeddings_input_dir = spark.sparkContext.getConf().get('spark.clr.embedding_input_dir')
    cluster_center_input_dir = spark.sparkContext.getConf().get('spark.clr.cluster_center_input_dir')
    writedir = spark.sparkContext.getConf().get('spark.clr.sense_associations_output_dir')

    # Read in DataFrames
    embeddings_df: DataFrame = spark.read.format("csv").option("header", True).load(embeddings_input_dir)
    cluster_center_df: DataFrame = spark.read.format("csv").option("header", True).load(cluster_center_input_dir)

    # Filter invalid embeddings
    embedding_valid_udf = F.udf(lambda s: embedding_valid(s), BooleanType())
    embeddings_df = embeddings_df.filter(embedding_valid_udf(embeddings_df[raw_embedding_col_name]))

    # Join with NLP results by lexeme
    df: DataFrame = embeddings_df.join(cluster_center_df,
                                       embeddings_df[lexeme_col_name] == cluster_center_df[lexeme_col_name])

    # Compare pairwise euclidean distance to generate a score
    euclid_distance_udf = F.udf(lambda emb1, emb2: cos_sim(emb1, emb2), FloatType())
    df = df.select(
        embeddings_df[note_id_col_name],
        embeddings_df[lexeme_col_name],
        cluster_center_df[sense_id_col_name],
        euclid_distance_udf(embeddings_df[raw_embedding_col_name],
                            cluster_center_df[cluster_center_col_name]).alias(euclid_score_col_name)
    )

    # And select the minimum
    min_euclid_df = df.groupBy(df[note_id_col_name],
                               df[lexeme_col_name]).agg(F.min(df[euclid_score_col_name])).alias(euclid_score_col_name)

    df = df.join(min_euclid_df,
                 (df[note_id_col_name] == min_euclid_df[note_id_col_name]) &
                 (df[lexeme_col_name] == min_euclid_df[lexeme_col_name]) &
                 (df[euclid_score_col_name] == min_euclid_df[euclid_score_col_name])
                 ).select(df[note_id_col_name],
                          df[lexeme_col_name],
                          df[sense_id_col_name])

    df.write.csv(path=writedir, mode="overwrite", header=True)


