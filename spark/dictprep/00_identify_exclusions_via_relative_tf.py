"""
Defines a set of normalized lexical forms to retain for analysis via frequency threshold filtering via the assumption
that if it is ~top 3% occurring (mean + 2 std deviations) in terms of frequency in the corpus, it is likely noise
that should be filtered

Required spark parameter: spark.clr.keep_output_dir - where to write resulting terms to keep as a CSV

"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame

from clinicallanguageresource.dictprep.site_modify import sparkutils, nlpio
from clinicallanguageresource.dictprep.site_modify.column_names import *

if __name__ == "__main__":
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-TF-Cleanup")
    writedir = spark.sparkContext.getConf().get('spark.clr.keep_output_dir')

    df: DataFrame = nlpio.get_nlp_artifact_table(spark)
    # First, get frequencies
    normalization_udf = F.udf(lambda lexeme, cui: nlpio.normalize_lexeme_concept(lexeme, cui))
    df = df.withColumn(lexeme_col_name, normalization_udf(df[lexical_variant_col_name], df[concept_id_col_name]))
    df = df.groupBy(df[lexeme_col_name]).agg(F.sum(F.lit(1)).alias(lexeme_count_col_name)).select(
        df[lexeme_col_name],
        F.col(lexeme_count_col_name)
    ).persist()
    df_stats = df.select(
        F.mean(df[lexeme_count_col_name]).alias(lexeme_count_avg),
        F.stddev(df[lexeme_count_col_name]).alias(lexeme_count_stddev)
    ).collect()
    mean: float = float(df_stats[0][lexeme_count_avg])
    stddev: float = float(df_stats[0][lexeme_count_stddev])
    cutoff: float = mean + 2 * stddev
    print("Mean:", mean, "StdDev:", stddev, "Cutoff:", cutoff)
    df = df.filter(df[lexeme_count_col_name] < F.lit(cutoff)).repartition(1)
    df.write.csv(path=writedir, mode="overwrite", header=True)
