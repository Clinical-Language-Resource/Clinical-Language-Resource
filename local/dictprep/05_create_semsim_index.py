import base64
import os
import sys

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from numpy import ndarray
from pandas import DataFrame

from clinicallanguageresource.dictprep.site_modify.column_names import *


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
    args = sys.argv[1:]
    # Setup args
    input_csv_path = args[0]
    index_output_dir = args[1]
    labels_output_dir = args[2]

    df: DataFrame = pd.read_csv(input_csv_path, header=0)
    df[label_col_name] = np.arange(len(df))

    # Now build the index
    index: AnnoyIndex = AnnoyIndex(768, "euclidean")
    cluster_center_col_idx = df.columns.get_loc(cluster_center_col_name)
    label_col_idx = df.columns.get_loc(label_col_name)
    for row in df.iterrows():
        emb: ndarray = np.frombuffer(base64.b64decode(row[1][cluster_center_col_idx]), dtype="float32")
        if emb.shape[0] != 768:
            emb = np.frombuffer(base64.b64decode(row[1][cluster_center_col_idx]), dtype="float64")
            if emb.shape[0] != 768:
                raise Exception("Improper dimensionality for input embedding:", emb.shape[0])
        # noinspection PyTypeChecker
        index.add_item(row[1][label_col_idx], emb.tolist())
    index.build(n_trees=1)  # Index is small enough for single tree/no sharding for optimal performance
    index.save(index_output_path)

    df = df[[label_col_name, lexeme_col_name, sense_id_col_name, lexeme_count_col_name, cluster_size_col_name]]
    df.to_csv(labels_output_path, header=True, index=False)







