"""
Determines semantic similarity between lexemes in the NLP dataset, thus creating a ranked list of potential synonyms
for use in NLP ruleset construction, etc.

In order to factor in potential sense disambiguations into semantic similarity computations, particularly to account
for lexemes potentially being associated with multiple senses, completion of 02_associate_nlp_with_senses.py is
recommended, but not required. If this is not done, all lexeme occurrences will be assumed to be from a single sense
with ID -1, which will result in the semantic similarity lookup being still generally-usable, but results from minority
occurrence senses will be drowned out for a given lexeme

To determine semantic similarity , uses the Normalized Google Distance (NGD) to measure the semantic similarity between
individual (lexeme, sense) tuples, grouping on a per-document basis to define "documents".

for a (lexeme, sense) x and another (lexeme, sense) y, the NGD-defined semantic similarity between the two is

NGD(x,y) = (max(log(f(x)), log(f(y))) - log(f(x,y)))/(log(N) - min(log(f(x)), log(f(y)))

Where f(x) represents the number of documents with (lexeme, sense) tuple x, and f(x, y) represents the number of
documents with both (lexeme, sense) tuples x and y


Required spark parameters:
    1) spark.clr.sense_associations_input_dir - Sense associations from step 2.
    2) spark.clr.semsim_output_dir - Where to write semantic similarity search results
    3) spark.sql.crossJoin.enabled - must be set to true as we use cross-joins to create x,y pairs

If 02_associate_nlp_with_senses is not ran, do not supply the spark.clr.sense_associations_input_dir parameter.
Instead, the steps from "Setup NLP Dataset" in 00_generate_embeddings_from_nlp_artifacts will be done instead
"""
from typing import Collection

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType

from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

import pyspark.sql.functions as F
import math


def num_shared_documents(set1: Collection, set2: Collection) -> int:
    return len(set(set1).intersection(set(set2)))


def calculate_ngd_score(x: int, y: int, xy: int, n: int) -> float:
    """
    Calculates normalized google distance between x and y
    NGD(x,y) = (max(log(f(x)), log(f(y))) - log(f(x,y)))/(log(N) - min(log(f(x)), log(f(y)))
    :param x: f(x) - # of documents containing x
    :param y: f(y) - # of documents containing y
    :param xy: f(x,y) - # of documents containing both x and y
    :param n: # of documents in corpus
    :return: NGD(x,y)
    """

    return (max(math.log(x), math.log(y)) - math.log(xy)) / (math.log(n) - min(math.log(x), math.log(y)))


if __name__ == "__main__":
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Find-Semantic-Similarities")
    writedir = spark.sparkContext.getConf().get('spark.clr.semsim_output_dir')

    df: DataFrame
    if spark.sparkContext.getConf().contains('spark.clr.sense_associations_input_dir'):
        sense_associations_input_dir = spark.sparkContext.getConf().get('spark.clr.sense_associations_input_dir')
        # Read in sense associations
        df: DataFrame = spark.read.format("csv").option("header", True).load(sense_associations_input_dir)
    else:
        df = nlpio.get_nlp_artifact_table(spark)
        df = nlpio.get_eligible_nlp_artifacts(df)
        df = nlpio.remove_all_subsumed(df)
        df = df.select(df[note_id_col_name],
                       F.lower(df[lexeme_col_name]).alias(lexeme_col_name)).distinct()
        df = df.select(df[note_id_col_name],
                       df[lexeme_col_name],
                       F.lit(-1).alias(sense_id_col_name)).persist()

    # Construct background dataframe
    backgroundDf: DataFrame = df.withColumn(lexeme_count_col_name, F.lit(1)) \
        .groupBy(df[lexeme_col_name]) \
        .agg(F.sum(F.col(lexeme_count_col_name)).alias(lexeme_count_col_name))

    # Parameter for NGD/GND
    doc_count: float = float(int(backgroundDf.agg(F.sum(backgroundDf[lexeme_count_col_name])).collect()[0][0]))

    # Code section below extensively uses cross-joins to generate permutations for testing
    # We don't cross-join on the same lexeme even if sense different under assumption that
    # the senses are mutually exclusive. Even if they are not, this would serve to magnify actual items that can be
    # used to construct sense definition, rather than outputting the fact that senses might be overlapping.
    # Even though this is less efficient here/does duplicate work, also compute the reverse equivalent x,y for
    # efficiency later on down the pipeline

    # Get document set for all (lexeme, sense) tokens
    term_documents_df = df.groupBy(
        df[lexeme_col_name],
        df[sense_id_col_name]
    ).agg(
        F.collect_set(df[note_id_col_name]).alias(document_set_col_name)
    ).persist()

    # f(x)
    term_freq = term_documents_df.select(
        term_documents_df[lexeme_col_name],
        term_documents_df[sense_id_col_name],
        F.size(term_documents_df[document_set_col_name]).alias(term_freq_col_name)
    )

    # f(x, y)
    f_x_y_udf = F.udf(lambda x, y: num_shared_documents(x, y))
    term_documents_df_2 = term_documents_df
    term_freq_combined = term_documents_df.join(
        term_documents_df_2,
        term_documents_df[lexeme_col_name] != term_documents_df_2[lexeme_col_name]
    ).select(
        term_documents_df[lexeme_col_name],
        term_documents_df[sense_id_col_name],
        term_documents_df_2[lexeme_col_name].alias(lexeme_col_name_2),
        term_documents_df_2[sense_id_col_name].alias(sense_id_col_name_2),
        f_x_y_udf(term_documents_df[document_set_col_name],
                  term_documents_df_2[document_set_col_name]).alias(combined_freq_col_name)
    )

    # Get all columns required for NGD
    # - First join x
    ngd_df = term_freq_combined.join(
        term_freq,
        (term_freq_combined[lexeme_col_name] == term_freq[lexeme_col_name]) & (
                term_freq_combined[sense_id_col_name] == term_freq[sense_id_col_name])
    ).select(
        term_freq_combined[lexeme_col_name],
        term_freq_combined[sense_id_col_name],
        term_freq_combined[lexeme_col_name_2],
        term_freq_combined[sense_id_col_name_2],
        term_freq_combined[combined_freq_col_name].alias(fxy_col_name),
        term_freq[term_freq_col_name].alias(fx_col_name)
    )
    # - Now join y
    ngd_df = ngd_df.join(
        term_freq,
        (ngd_df[lexeme_col_name_2] == term_freq[lexeme_col_name]) & (
                ngd_df[sense_id_col_name_2] == term_freq[sense_id_col_name])
    ).select(
        ngd_df[lexeme_col_name],
        ngd_df[sense_id_col_name],
        ngd_df[lexeme_col_name_2],
        ngd_df[sense_id_col_name_2],
        ngd_df[fxy_col_name],
        ngd_df[fx_col_name],
        term_freq[term_freq_col_name].alias(fy_col_name)
    )

    # Now actually calculate NGD and save
    ngd_udf = F.udf(lambda x, y, xy, n: calculate_ngd_score(x, y, xy, n), FloatType())
    ngd_df = ngd_df.select(
        ngd_df[lexeme_col_name],
        ngd_df[sense_id_col_name],
        ngd_df[lexeme_col_name_2],
        ngd_df[sense_id_col_name_2],
        ngd_udf(
            ngd_df[fx_col_name],
            ngd_df[fy_col_name],
            ngd_df[fxy_col_name],
            F.lit(doc_count)
        )
    )

    ngd_df.write.csv(path=writedir, mode="overwrite", header=True)
