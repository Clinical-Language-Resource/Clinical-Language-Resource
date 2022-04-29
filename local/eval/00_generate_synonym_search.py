import argparse
import base64
import re

import numpy as np
import pandas as pd
import torch
from annoy import AnnoyIndex
from numpy import ndarray
from pandas import DataFrame
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, AutoModel


def generate_embedding(lexeme: str, sentence: str):
    torch.set_num_threads(1)  # Hard-lock to single thread since we are running many partitions in parallel
    layers = [-4, -3, -2, -1]  # Use last 4 layers by default as is standard

    # Find possible lexeme indices within the sentence. Note that the sentence can contain lexemes multiple times,
    # but because we eliminate duplicates from offsets, we will only have one entry. Thus, we need to iterate through
    # all of them
    lexeme_offsets = [m.start() for m in re.finditer(re.escape(lexeme), sentence)]

    ret_vectors = []

    # Generate a separate embedding for each instance
    for lexeme_offset in lexeme_offsets:
        # Tokenize to BERT format and identify indexes/subwords corresponding to original word of interest
        lexeme_end = lexeme_offset + len(lexeme)

        try:
            encoded = tokenizer.encode_plus(sentence, return_tensors="pt")
            encoded_token_idxs = []
            bert_tokenization_offsets = encoded.encodings[0].offsets
            for i in range(0, len(bert_tokenization_offsets)):
                encoded_token_pos = bert_tokenization_offsets[i]
                start = encoded_token_pos[0]
                end = encoded_token_pos[1]
                if start == 0 and end == 0:
                    continue  # [CLS], [SEP], etc.
                # Collision check
                if (start <= lexeme_offset and end > lexeme_offset) or (start >= lexeme_offset and start < lexeme_end):
                    encoded_token_idxs.append(i)
            # Now get the actual embeddings from hidden layer - average subword embeddings to generate representation
            # for lexeme as a whole
            with torch.no_grad():
                output = model(**encoded)
            states = output.hidden_states
            output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
            word_tokens_output = output[encoded_token_idxs].mean(dim=0)
            npemb: np.ndarray = np.array(word_tokens_output)
            # Should not happen, but just in case
            if np.any(np.isnan(npemb)) or not np.all(np.isfinite(npemb)):
                continue
            encoded = base64.b64encode(npemb.tobytes()).decode('ascii')
            ret_vectors.append(encoded)
        except Exception as e:
            print(e)
            continue
    return ret_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('search_terms', type=str)
    parser.add_argument('results_output', type=str)
    args = parser.parse_args()

    df: DataFrame = pd.read_csv('labels_all.csv', index_col=["label"])

    # Now build the index
    with open(args.search_terms, 'r') as f:
        with open(args.results_output, 'w') as f2:
            f2.write("search_term\tsem_type\tranking\tresult_lexeme\tresult_sense_id\n")
            for raw_text in f.readlines():
                parameterized_text = raw_text.lower().strip()
                text = parameterized_text.split("|")[0]
                context = parameterized_text.split("|")[1]
                gensentence = "Patient presented today with " + context + " associated " + text
                tokenizer: BertTokenizer = AutoTokenizer.from_pretrained("./models/bio_clinbert_model",
                                                                         config=AutoConfig.from_pretrained(
                                                                             "./models/bio_clinbert_model"))
                model = AutoModel.from_pretrained("./models/bio_clinbert_model", output_hidden_states=True)

                npemb: str = generate_embedding(text, gensentence)[0]

                emb: ndarray = np.frombuffer(base64.b64decode(npemb), dtype="float32")
                if emb.shape[0] != 768:
                    emb = np.frombuffer(base64.b64decode(npemb), dtype="float64")
                    if emb.shape[0] != 768:
                        raise Exception("Improper dimensionality for input embedding:", emb.shape[0])

                for prefix in ["DISO", "PROC", "FIND", "DRUG", "ANAT", "ALL"]:
                    index: AnnoyIndex = AnnoyIndex(768, 'euclidean')
                    index.load(prefix + '_index.ann')
                    similar = index.get_nns_by_vector(emb.tolist(), 20)
                    i = 0
                    for idx in similar:
                        i += 1
                        f2.write(
                            text + "\t"
                            + prefix + "\t"
                            + str(i) + "\t"
                            + str(df['lexeme'][idx]).replace("\t", " ") + "\t"
                            + str(df['sense_id'][idx]) + "\n")
                        print(
                            text + "\t"
                            + prefix + "\t"
                            + str(i) + "\t"
                            + str(df['lexeme'][idx]).replace("\t", " ") + "\t"
                            + str(df['sense_id'][idx]) + "\n")
                break



