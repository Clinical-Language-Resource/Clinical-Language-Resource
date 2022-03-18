# Generates Embeddings from NLP Artifacts
import base64
import re

import numpy as np
import torch
from pyspark.sql import SparkSession, functions, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AutoConfig

import clinicallanguageresource.dictprep.site_modify.nlpio as nlpio
from clinicallanguageresource.dictprep.site_modify import sparkutils


def generate_embedding(lexeme: str, sentence: str):
    torch.set_num_threads(1)  # TODO make torch settings configurable instead of being locked to CPU single-thread
    layers = [-4, -3, -2, -1]  # Use last 4 layers by default as is standard

    # Find possible lexeme indices within the sentence. Note that the sentence can contain lexemes multiple times,
    # but because we eliminate duplicates from offsets, we will only have one entry. Thus, we need to iterate through
    # all of them
    lexeme_offsets = [m.start() for m in re.finditer(lexeme, sentence)]

    ret_vectors = []

    # Generate a separate embedding for each instance
    for lexeme_offset in lexeme_offsets:
        # Tokenize to BERT format and identify indexes/subwords corresponding to original word of interest
        # We can't do a word-by-word lookup due to multi-word lexemes and the sequence being meaningful, so instead
        # find whitespace offset position and add based on lexeme token length
        lexeme_token_length = len(lexeme.split(" "))
        token_idxes = []
        for i in range(0, lexeme_token_length):
            token_idxes.append(i + lexeme_offset)
        try:
            encoded = tokenizer.encode_plus(sentence, return_tensors="pt")
            encoded_token_idxs = np.where(np.isin(np.array(encoded.word_ids()), token_idxes))
            # Now get the actual embeddings from hidden layer - average subword embeddings to generate representation
            # for lexeme as a whole
            with torch.no_grad():
                output = model(**encoded)
            states = output.hidden_states
            output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
            word_tokens_output = output[encoded_token_idxs].mean(dim=0)
            # Convert to base64 for ease of export (and we can always convert back for mild performance cost anyways)
            # Does cost more in storage space though
            ret_vectors.append(base64.b64encode(np.array(word_tokens_output).tobytes()).decode('ascii'))
        except Exception:
            continue
    return ret_vectors




if __name__ == '__main__':
    # Setup Spark
    spark: SparkSession = sparkutils.setup_spark_session("CLR-Generate-Embeddings")
    save_embeddings_dir = spark.sparkContext.getConf().get('spark.clr.embedding_output_dir')

    # Setup NLP Dataset
    df: DataFrame = nlpio.get_nlp_artifact_table(spark)
    df = nlpio.get_eligible_nlp_artifacts(df)
    df = nlpio.remove_all_subsumed(df)
    df = df.select(df[nlpio.note_id_col_name],
                   df[nlpio.concept_code_col_name],
                   functions.lower(df[nlpio.containing_sentence_col_name]).alias(nlpio.containing_sentence_col_name),
                   functions.lower(df[nlpio.lexeme_col_name]).alias(nlpio.lexeme_col_name)).distinct()

    # Setup BERT
    tokenizer: BertTokenizer = AutoTokenizer.from_pretrained("./model/bio_clinbert_tokenizer",
                                                             config=AutoConfig.from_pretrained("./model/bio_clinbert_model"))
    model = AutoModel.from_pretrained("./model/bio_clinbert_model", output_hidden_states=True)
    torch.set_num_threads(1)

    # Run embedding generation
    emb_func_output_type = ArrayType(StringType())
    embeddings_udf = F.udf(lambda lex, sent: generate_embedding(lex, sent), emb_func_output_type)
    df = df.select(df[nlpio.note_id_col_name],
                   df[nlpio.concept_code_col_name],
                   df[nlpio.lexeme_col_name],
                   functions.explode(embeddings_udf(df[nlpio.lexeme_col_name],
                                                    df[nlpio.containing_sentence_col_name])).alias("embedding"))

    # Save results as CSV
    df.write.csv(path=save_embeddings_dir, mode="overwrite", header=True)


