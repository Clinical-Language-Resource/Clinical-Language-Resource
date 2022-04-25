import base64
import os
import sys

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from numpy import ndarray
from pandas import DataFrame

from clinicallanguageresource.dictprep.site_modify.column_names import *

SUPPORTED_TYPES = ["DISO", "PROC", "FIND", "DRUG", "ANAT", "ALL"]

if __name__ == "__main__":
    args = sys.argv[1:]
    # Setup args
    input_csv_path = args[0]
    index_output_dir = args[1]
    labels_output_path = args[2]
    cui_mappings_path = args[3]
    semtype_mappings_path = args[4]

    cluster_centers: DataFrame = pd.read_csv(input_csv_path, header=0)
    cluster_centers[label_col_name] = np.arange(len(cluster_centers))

    cui_mappings = pd.read_csv(cui_mappings_path, header=None, delimiter='|', names=["CUI"])['CUI']
    semtypes = pd.read_csv(semtype_mappings_path, header=None, delimiter='|', names=["SEMTYPE"])['SEMTYPE']

    # Now build the index
    index_dict = {
    }
    for type in SUPPORTED_TYPES:
        index: AnnoyIndex = AnnoyIndex(768, "euclidean")
        index_dict[type] = index
    cluster_center_col_idx = cluster_centers.columns.get_loc(cluster_center_col_name)
    label_col_idx = cluster_centers.columns.get_loc(label_col_name)
    for row in cluster_centers.iterrows():
        emb: ndarray = np.frombuffer(base64.b64decode(row[1][cluster_center_col_idx]), dtype="float32")
        if emb.shape[0] != 768:
            emb = np.frombuffer(base64.b64decode(row[1][cluster_center_col_idx]), dtype="float64")
            if emb.shape[0] != 768:
                raise Exception("Improper dimensionality for input embedding:", emb.shape[0])
        # Determine where to output
        semtypes_out = ["ALL"]
        lexeme = row[1][lexeme_col_name]
        for cui in cui_mappings.loc[[lexeme.lower()]]:
            for semtype in semtypes.loc[[cui]]:
                semtypes_out.append(semtype)
        # And now output
        for semtype in set(semtypes_out):
            if semtype not in SUPPORTED_TYPES:
                continue
            # noinspection PyTypeChecker
            index_dict[semtype].add_item(row[1][label_col_idx], emb.tolist())
    for semtype in index_dict:
        index_dict[semtype].build(n_trees=1)  # Index is small enough for single tree/no sharding for performance
        index_dict[semtype].save(semtype + "_index.ann")

    cluster_centers = cluster_centers[
        [label_col_name, lexeme_col_name, sense_id_col_name, lexeme_count_col_name, cluster_size_col_name]
    ]
    cluster_centers.to_csv(labels_output_path, header=True, index=False)







