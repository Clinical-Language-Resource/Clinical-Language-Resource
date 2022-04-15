import base64
import sys

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from numpy import ndarray
from pandas import DataFrame

from clinicallanguageresource.dictprep.site_modify.column_names import *

if __name__ == "__main__":
    args = sys.argv[1:]
    # Setup Spark
    input_csv_path = args[0]
    index_output_path = args[1]
    labels_output_path = args[2]

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







