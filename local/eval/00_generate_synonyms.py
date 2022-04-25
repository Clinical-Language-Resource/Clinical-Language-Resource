import pandas as pd
from annoy import AnnoyIndex
from pandas import DataFrame


def print_nns_to_string(dict_idx, ann_idx, src_df):
    vals = map(lambda idx: src_df.loc[src_df['label'] == idx]['lexeme'][idx], ann_idx.get_nns_by_item(dict_idx, 10))
    vals = map(lambda lexeme: lexeme.replace("\t", ' ').lower(), vals)
    return ','.join(vals)


if __name__ == "__main__":
    df: DataFrame = pd.read_csv('labels_all.csv', header=0)

    index: AnnoyIndex = AnnoyIndex(768, 'euclidean')
    index.load('index_all.idx')

    df['sem_sim'] = df.apply(lambda row: print_nns_to_string(row['label'], index, df), axis=1)

    df.to_csv('semsim_all.csv', header=True, index=False)



