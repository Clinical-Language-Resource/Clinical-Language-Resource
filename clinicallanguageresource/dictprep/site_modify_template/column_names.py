"""
Contains a list of column names used by this application
"""

# Direct I/O Columns. These are based on OHDSI CDM note_nlp and may be needed to be modified as appropriate
note_id_col_name = "note_id"
containing_sentence_col_name = "snippet"
lexical_variant_col_name = "lexical_variant"
concept_id_col_name = "note_nlp_concept_id"
offset_col_name = "offset"

# Column names used internally. Only need to change this if name conflicts with an existing column in your import
concept_code_col_name = "concept"
lexeme_col_name = "lexeme"
begin_col_name = "begin"
end_col_name = "end"
lexeme_index_column_name = "lexeme_indexes"
lexeme_extract_struct_name = "lstc"
raw_embedding_col_name = "embedding"
lexeme_count_col_name = "lexeme_count"
cluster_center_col_name = "cluster_center"
sense_id_col_name = "sense_id"
lexeme_sample_idx_col_name = "lexeme_sample_idx"
cluster_info_struct_name = "cluster_info"
cluster_size_col_name = "cluster_size"
