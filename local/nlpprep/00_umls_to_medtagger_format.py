"""
Converts a UMLS Installation to MedTagger compatible dictionary format.
Note that this is not immediately usable - MedTagger's dictionary preparation pipeline must first be
run to LVG'ize the entries before it can be used as a MedTagger dictionary

A spark alternative to this script is also available under the spark package
"""
import os.path
import sys

import pandas as pd
from pandas import DataFrame

# Configuration Settings
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
    "T019", "T020", "T037", "T047", "T048", "T049", "T050", "T190", "T191"
]

FIND = [
    "T033", "T034", "T040", "T041", "T042", "T043", "T044", "T045", "T046", "T056", "T057", "T184"
]

PROC = ["T059", "T060", "T061"]

ANAT = ["T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030"]

if __name__ == "__main__":
    args = sys.argv[1:]
    umls_installation_path = args[0]
    dict_output_path = args[1]

    # Read in UMLS RRF files
    mrconso_df: DataFrame = pd.read_csv(
        os.path.join(umls_installation_path, "META", "MRCONSO.RRF"),
        header=None,
        delimiter='|',
        names=['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF'],
        low_memory=False,
        index_col=False,
        dtype='str'
    )

    mrsty_df: DataFrame = pd.read_csv(
        os.path.join(umls_installation_path, "META", "MRSTY.RRF"),
        header=None,
        delimiter='|',
        names=['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF'],
        low_memory=False,
        index_col=False,
        dtype='str'
    )

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
    mrsty_df['SEMANTIC_TYPE'] = mrsty_df.apply(
        lambda row: mappings[row['TUI']] if row['TUI'] in mappings.keys() else None,
        axis=1
    )
    mrsty_df = mrsty_df[mrsty_df['SEMANTIC_TYPE'].notnull()]
    # - Now condense so that there is only one row per CUI by merging the semantic_types
    mrsty_df = mrsty_df.groupby(['CUI'])['SEMANTIC_TYPE'].apply(set).reset_index(name='SEMANTIC_TYPE_SET')
    mrsty_df['SEMANTIC_TYPES'] = mrsty_df.apply(
        lambda row: ';'.join(row['SEMANTIC_TYPE_SET']),
        axis=1
    )
    mrsty_df = mrsty_df[['CUI', 'SEMANTIC_TYPES']]
    # Join with MRCONSO to get strings, and format to match MedTagger format
    # - Apply inclusion filters
    if len(LANGS) > 0:
        mrconso_df['LAT_FILTER'] = mrconso_df.apply(
            lambda row: row['LAT'] in LANGS,
            axis=1
        )
        mrconso_df = mrconso_df[mrconso_df['LAT_FILTER']]
    if len(VOCABS) > 0:
        mrconso_df['SAB_FILTER'] = mrconso_df.apply(
            lambda row: row['SAB'] in VOCABS,
            axis=1
        )
        mrconso_df = mrconso_df[mrconso_df['SAB_FILTER']]
    # noinspection PyTypeChecker
    df: DataFrame = mrconso_df.merge(mrsty_df, on='CUI', how='inner', suffixes=(None, "_sty"))
    # - Select in MedTagger format
    df = df[['STR', 'CUI', 'SEMANTIC_TYPES']].drop_duplicates()
    df.to_csv(dict_output_path, sep='|', header=False, index=False)
