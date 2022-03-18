"""
Handles interfacing with NLP result tables containing UMLS dictionary lookup records.
For simplicity, the provided implementation here assumes OHDSI CDM format for note_nlp, but settings here can be
adjusted based on use case and needs, with individual methods also overloaded as appropriate
"""

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from clinicallanguageresource.dictprep.util.nlpannotations import flatten_overlaps, flatten_overlaps_schema

# Column names assume OHDSI CDM note_nlp table format, adjust accordingly for your dataset
note_id_col_name = "note_id"
containing_sentence_col_name = "snippet"
lexical_variant_col_name = "lexical_variant"
concept_id_col_name = "note_nlp_concept_id"
offset_col_name = "offset"

# Column names used internally. Only need to change this if name conflicts with an existing column in your import
concept_code_col_name = "concept"
lexeme_col_name = "lexeme"
begin_col_name = "begin"
end_col_name = "end"


def get_nlp_artifact_table(spark: SparkSession) -> DataFrame:
    """
    Retrieves a nlp artifact table from hive

    :param spark: The spark session
    :return: A DataFrame constructed from the table
    """
    return spark.read.format("hive").table("nlp_results_table")


def get_eligible_nlp_artifacts(df: DataFrame) -> DataFrame:
    """
    Filters results from :func:`get_nlp_artifact_table` to only records that are positive, present, and pertaining
    to the patient. Depending on how get_nlp_artifact_table is defined, it may be appropriate to simply return
    the same dataframe without additional filtering

    :param df: The NLP Artifact Table
    :return: A filtered version of the NLP Artifact Table
    """
    return df.filter(df['term_exists'] == "Y")


def remove_all_subsumed(df: DataFrame) -> DataFrame:
    """
    For overlapping spans, keeps only the longest. Can be changed to noop if NLP engine
    handles this already

    :param df: The NLP Artifact table
    :return: The filtered dataframe with all subsumed removed with columns note_id, concept_code, sentence, lexeme
    """
    df = df.groupBy(
        df[note_id_col_name],
        df[containing_sentence_col_name]) \
        .agg(F.collect_list(F.struct(df[lexical_variant_col_name], df[offset_col_name], df[concept_id_col_name]))
             .alias("lexeme_indexes"))
    # Remove all subsumed annotations
    remove_subsumed_udf = F.udf(lambda offsets: flatten_overlaps(offsets), flatten_overlaps_schema(concept_code_col_name,
                                                                                                 lexeme_col_name,
                                                                                                 begin_col_name,
                                                                                                 end_col_name))
    df = df.select(df[note_id_col_name], df[containing_sentence_col_name],
                   F.explode(remove_subsumed_udf(df[containing_sentence_col_name], df["lexeme_indexes"])).alias("lstc"))
    # Keep only note_id, concept_code, sentence, lexeme, and deduplicate
    df = df.select(df[note_id_col_name],
                   F.col("lstc." + concept_code_col_name).alias(concept_code_col_name),
                   df[containing_sentence_col_name],
                   F.col("lstc." + lexeme_col_name).alias(lexeme_col_name))
    return df
