import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType
from pyspark.sql.window import Window


from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

if __name__ == "__main__":
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Find-Embedding-Centers")
    embeddings_input_dir = spark.sparkContext.getConf().get('spark.clr.embedding_input_dir')
    max_wsd_clusters = int(spark.sparkContext.getConf().get('spark.clr.max_wsd_clusters'))
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
        normalize_lexeme_udf(sense_counts[lexical_variant_col_name], sense_counts[concept_id_col_name]).alias(lexeme_col_name),
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
    df = df.select(df[lexeme_col_name], df[raw_embedding_col_name], F.lit(1).alias(lexeme_count_col_name))\
        .groupBy(df[lexeme_col_name])\
        .agg(F.collect_list(df[raw_embedding_col_name]).alias(raw_embedding_col_name),
             F.sum(F.col(lexeme_count_col_name)).alias(lexeme_count_col_name))
    if tl_filter > 0:
        df = df.filter(F.length(df[lexeme_col_name]) >= tl_filter)


