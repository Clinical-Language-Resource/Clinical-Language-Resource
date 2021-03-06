"""
Generates a BERT embedding for each NLP artifact provided in an input table.
Modify clinicallanguageresource.dictprep.site_modify appropriately to your cluster.
clinicallanguageresource.dictprep.site_modify_template provides a baseline implementation on the OHDSI CDM's
note_nlp table.

Requires prior completion of 00_identify_exclusions_via_relative_tf.py

Output will be in form of note_id, concept_code, lexeme, base64-encoded ndarray of the embedding

Required spark parameters:
1) spark.clr.embedding_output_dir - where to write resulting embeddings as a CSV
2) spark.clr.filter_input_dir - Where output from 00_identify_exclusions_via_relative_tf.py  was stored
"""

import base64
import re
import struct as st

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AutoConfig

from clinicallanguageresource.dictprep.site_modify.column_names import *
import clinicallanguageresource.dictprep.site_modify.nlpio as nlpio
from clinicallanguageresource.dictprep.site_modify import sparkutils


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
            encoded = base64.b64encode(npemb.tobytes()).decode('ascii');
            ret_vectors.append(encoded)
        except Exception as e:
            print(e)
            continue
    return ret_vectors


if __name__ == '__main__':
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Generate-Embeddings")
    save_embeddings_dir = spark.sparkContext.getConf().get('spark.clr.embedding_output_dir')
    filter_input_dir = spark.sparkContext.getConf().get('spark.clr.filter_input_dir')

    # Setup NLP Dataset
    filter_df: DataFrame = spark.read.format("csv").option("header", True).load(filter_input_dir)
    df: DataFrame = nlpio.get_nlp_artifact_table(spark)
    df = nlpio.get_eligible_nlp_artifacts(df)
    df = nlpio.remove_all_subsumed(df)

    # Filter out too-high-frequency base lexemes on assumption that it is not a legitimate match
    normalization_udf = F.udf(lambda lexeme, concept_code: nlpio.normalize_lexeme_concept(lexeme, concept_code),
                              StringType())
    df = df.join(
        filter_df,
        normalization_udf(df[lexeme_col_name], df[concept_code_col_name]) == filter_df[lexeme_col_name]
    ).select(df[note_id_col_name],
             df[concept_code_col_name],
             F.lower(df[containing_sentence_col_name]).alias(containing_sentence_col_name),
             F.lower(df[lexeme_col_name]).alias(lexeme_col_name)).distinct()

    # Setup BERT
    tokenizer: BertTokenizer = AutoTokenizer.from_pretrained("./model/bio_clinbert_model",
                                                             config=AutoConfig.from_pretrained(
                                                                 "./model/bio_clinbert_model"))
    model = AutoModel.from_pretrained("./model/bio_clinbert_model", output_hidden_states=True)
    torch.set_num_threads(1)  # Hard-lock to single thread since we are running many partitions in parallel

    # Run embedding generation
    embeddings_udf = F.udf(lambda lex, sent: generate_embedding(lex, sent), ArrayType(StringType()))

    # De-LVG-ize lexemes if necessary
    df = df.select(df[note_id_col_name],
                   df[concept_code_col_name],
                   normalization_udf(df[lexeme_col_name], df[concept_code_col_name]).alias(lexeme_col_name),
                   F.explode(embeddings_udf(df[lexeme_col_name],
                                            df[containing_sentence_col_name])).alias(raw_embedding_col_name))

    # Save results as CSV
    df.write.csv(path=save_embeddings_dir, mode="append", header=True)
