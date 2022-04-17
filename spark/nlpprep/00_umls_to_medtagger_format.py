"""
Converts a UMLS Installation to MedTagger compatible dictionary format.
Note that this is not immediately usable - MedTagger's dictionary preparation pipeline must first be
run to LVG'ize the entries before it can be used as a MedTagger dictionary

Alternative Spark version of the local run
"""
import sys

import pyspark.sql.functions as F
# Configuration Settings
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType

from clinicallanguageresource.dictprep.site_modify import sparkutils

LANGS = ['ENG']  # Languages to Retain using UMLS LAT codes. Set to empty to disable filter
VOCABS = [  # Source Vocabularies to Retain using UMLS versioned SAB codes. Set to empty to disable filter
    'SNOMEDCT_US_2021_09_01',
    'SNOMEDCT_US_2017_01_31',
    'SNOMEDCT_US_2018_03_01',
    'SNOMEDCT_US_2021_03_01',
    'RXNORM_20AA_210907F'
]

# Type mappings - slightly modified based on cTAKES SemanticUtil. Should not need any modification
DRUG = [
    "T109", "T110", "T114", "T115", "T116", "T118", "T119", "T121", "T122", "T123", "T124", "T125", "T126", "T127",
    "T129", "T130", "T131", "T195", "T196", "T197", "T200", "T203"
]

DISO = [
    "T019", "T020", "T037", "T046", "T047", "T048", "T049", "T050", "T190", "T191"
]

FIND = [
    "T032", "T033", "T034", "T039", "T040", "T041", "T042", "T043", "T044", "T045", "T051", "T052", "T053", "T054",
    "T055", "T056", "T057", "T184"
]

PROC = ["T058", "T059", "T060", "T061", "T062", "T063", "T065"]

ANAT = ["T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030"]

if __name__ == "__main__":

    # Read in UMLS RRF files
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Prep-MedTagger-Dict")
    mrconso_path = spark.sparkContext.getConf().get('spark.clr.mrconso_path')
    mrsty_path = spark.sparkContext.getConf().get('spark.clr.mrsty_path')
    output_path = spark.sparkContext.getConf().get('spark.clr.output_path')

    mrconso_schema: StructType = StructType([
        StructField("CUI", StringType(), True),
        StructField("LAT", StringType(), True),
        StructField("TS", StringType(), True),
        StructField("LUI", StringType(), True),
        StructField("STT", StringType(), True),
        StructField("SUI", StringType(), True),
        StructField("ISPREF", StringType(), True),
        StructField("AUI", StringType(), True),
        StructField("SAUI", StringType(), True),
        StructField("SCUI", StringType(), True),
        StructField("SDUI", StringType(), True),
        StructField("SAB", StringType(), True),
        StructField("TTY", StringType(), True),
        StructField("CODE", StringType(), True),
        StructField("STR", StringType(), True),
        StructField("SRL", StringType(), True),
        StructField("SUPPRESS", StringType(), True),
        StructField("CVF", StringType(), True)
    ])
    mrconso_df: DataFrame = spark.read.csv(mrconso_path, sep='|', header=False, schema=mrconso_schema).repartition(200)

    mrsty_schema: StructType = StructType([
        StructField("CUI", StringType(), True),
        StructField("TUI", StringType(), True),
        StructField("STN", StringType(), True),
        StructField("STY", StringType(), True),
        StructField("ATUI", StringType(), True),
        StructField("CVF", StringType(), True)
    ])
    mrsty_df: DataFrame = spark.read.csv(mrsty_path, sep='|', header=False, schema=mrsty_schema).repartition(200)

    # Map MRSTY to appropriate types
    # - First construct type mappings
    mappings: dict = {}
    for tui in DRUG:
        mappings[tui] = 'DRUG:' + tui
    for tui in DISO:
        mappings[tui] = 'DISO:' + tui
    for tui in FIND:
        mappings[tui] = 'FIND:' + tui
    for tui in PROC:
        mappings[tui] = 'PROC:' + tui
    for tui in ANAT:
        mappings[tui] = 'ANAT:' + tui
    # - Then filter mrsty to only retain the CUIs with these mappings
    map_type_udf = F.udf(lambda tui: mappings[tui] if tui in mappings.keys() else "", StringType())
    mrsty_df = mrsty_df.withColumn('SEMANTIC_TYPE', map_type_udf(mrsty_df['TUI']))
    mrsty_df = mrsty_df.filter(F.length(mrsty_df['SEMANTIC_TYPE']) > 0)
    # - Now condense so that there is only one row per CUI by merging the semantic_types
    mrsty_df = mrsty_df.groupby(mrsty_df['CUI'])\
        .agg(F.collect_set(mrsty_df['SEMANTIC_TYPE']).alias('SEMANTIC_TYPE_SET'))
    mrsty_df = mrsty_df.withColumn('SEMANTIC_TYPES', F.concat_ws(';', mrsty_df['SEMANTIC_TYPE_SET']))
    mrsty_df = mrsty_df.select('CUI', 'SEMANTIC_TYPES')
    # Join with MRCONSO to get strings, and format to match MedTagger format
    # - Apply inclusion filters
    if len(LANGS) > 0:
        filter_udf = F.udf(lambda lang: lang in LANGS, BooleanType())
        mrconso_df = mrconso_df.filter(filter_udf(mrconso_df['LAT']))
    if len(VOCABS) > 0:
        filter_udf = F.udf(lambda vocab: vocab in VOCABS, BooleanType())
        mrconso_df = mrconso_df.filter(filter_udf(mrconso_df['SAB']))
    df: DataFrame = mrconso_df.join(mrsty_df, mrconso_df['CUI'] == mrsty_df['CUI'])
    # - Select in MedTagger format
    df = df.select(mrconso_df['STR'], mrconso_df['SAB'], mrconso_df['CUI'], mrsty_df['SEMANTIC_TYPES']).distinct()
    df.coalesce(1).write.csv(path=output_path, header=None, sep='|')
