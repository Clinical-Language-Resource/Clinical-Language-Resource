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
lexeme_count_avg = "lexeme_avg"
lexeme_count_stddev = "lexeme_stddev"
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
euclid_score_col_name = "euclid_score"
lexeme_sense_struct_name = "lexeme_sense"
document_sense_col_name = "document"
term_freq_col_name = "term_freq"
combined_freq_col_name = "combined_freq"
lexeme_col_name_2 = lexeme_col_name + "_2"
sense_id_col_name_2 = sense_id_col_name + "_2"
fx_col_name = "fx"
fy_col_name = "fy"
fxy_col_name = "fxy"
ngd_col_name = "ngd"

