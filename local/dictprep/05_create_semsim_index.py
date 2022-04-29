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

    cui_mappings: DataFrame = pd.read_csv(cui_mappings_path, header=None, delimiter='|', names=["LEX", "CUI"],
                                          index_col=False)
    semtypes: DataFrame = pd.read_csv(semtype_mappings_path, header=None, delimiter='|', names=["CUI", "SEMTYPE"],
                                      index_col=False)

    # Determine Semtypes
    # - First join cuis to semtypes and generate a comprehensive list for each CUI
    cui_mappings["LEX"] = cui_mappings["LEX"].apply(lambda s: str(s).upper())
    merged_mappings = cui_mappings.merge(semtypes, on="CUI")
    def to_set(x):
        return set(x)
    merged_mappings = merged_mappings.groupby("LEX")["SEMTYPE"].agg(SEMTYPES=to_set)
    # - Now map to semtypes
    cluster_centers = cluster_centers.merge(merged_mappings, left_on=lexeme_col_name, right_on="LEX")

    # Now build the index
    index_dict = {
    }
    for type in SUPPORTED_TYPES:
        index: AnnoyIndex = AnnoyIndex(768, "euclidean")
        index_dict[type] = index

    # - Get Numpy Embedding Equivalent
    def get_numpy_embedding(numpystr):
        emb: ndarray = np.frombuffer(base64.b64decode(numpystr), dtype="float32")
        if emb.shape[0] != 768:
            emb = np.frombuffer(base64.b64decode(numpystr), dtype="float64")
            if emb.shape[0] != 768:
                raise Exception("Improper dimensionality for input embedding:", emb.shape[0])
        return emb

    # - And replace the base64 equivalent
    cluster_centers[cluster_center_col_name] = cluster_centers[cluster_center_col_name].apply(
        lambda e: get_numpy_embedding(e)
    )

    # - Now populate indices
    def index_embeddings(label, emb, semtypes, output_dict):
        for semtype in semtypes:
            if semtype not in SUPPORTED_TYPES:
                continue
            # noinspection PyTypeChecker
            output_dict[semtype].add_item(label, emb.tolist())
        output_dict["ALL"].add_item(label, emb.tolist())

    cluster_centers.apply(
        lambda r: index_embeddings(
            r[label_col_name],
            r[cluster_center_col_name],
            r["SEMTYPES"],
            index_dict
        ),
        axis=1
    )

    # - And Write out the final index
    for semtype in index_dict:
        index_dict[semtype].build(n_trees=1)  # Index is small enough for single tree/no sharding for performance
        index_dict[semtype].save(semtype + "_index.ann")

    cluster_centers = cluster_centers[
        [label_col_name, lexeme_col_name, sense_id_col_name, lexeme_count_col_name, cluster_size_col_name]
    ]
    cluster_centers.to_csv(labels_output_path, header=True, index=False)







