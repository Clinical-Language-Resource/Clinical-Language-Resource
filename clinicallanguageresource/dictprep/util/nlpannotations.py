from typing import List, Tuple

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DataType


def flatten_overlaps_longest(
        lexeme_offsets: List[Tuple[object, object, object]]
) -> List[Tuple[str, str, int, int]]:
    """
    Returns only the longest, non-overlapping, spans. The assumption is made that the lexemes and associated offsets
    all originate from the same sentence

    :param lexeme_offsets: A list of lexemes found in the sentence. Format should be a tuple of the lexeme itself,
     its character offset (can be either within sentence or document, as long as it is consistent),
     and the concept code to which it corresponds, and its semantic types
    :return: A list of concept code, lexeme, begin, end tuples consisting of only the longest distinct (non-overlapping)
     lexemes
    """
    # Convert lexeme_offsets into length, start, end, lexeme tuples and sort by descending length.
    offset_tuples: List[Tuple[int, int, int, str, str]] = []
    max_len = 0
    for lexeme_offset in lexeme_offsets:
        length: int = len(str(lexeme_offset[0]))
        start: int = int(str(lexeme_offset[1]))  # handle cases where input could be either a string or int
        end: int = start + length
        max_len = max(max_len, end)
        offset_tuples.append((length, start, end, str(lexeme_offset[0]), str(lexeme_offset[2])))
    offset_tuples.sort(key=lambda t: t[0], reverse=True)
    # Now iterate through the list in descending order and populate already visited indices. If a subsequent
    # offset index is already populated, then that lexeme is subsumed and should be excluded
    sentence_occupied = [0] * max_len
    output_offsets = []
    for offset in offset_tuples:
        begin: int = offset[1]
        end: int = offset[2]
        lexeme: str = offset[3]
        concept_code: str = offset[4]
        write: bool = True
        for i in range(begin, end):
            if sentence_occupied[i] == 1:
                write = False
                break
        if write:
            for i in range(begin, end):
                sentence_occupied[i] = 1
            output_offsets.append((concept_code, lexeme, begin, end))
    return output_offsets


def flatten_overlaps_atomic(
        lexeme_offsets: List[Tuple[object, object, object, object]]
) -> List[Tuple[str, str, int, int]]:
    """
    Returns only the shortest, atomic spans for each semantic type. The assumption is made that the lexemes and
    associated offsets all originate from the same sentence. If the annotation covers another annotation, the covering
    annotation is removed if at least one of the covered annotations share the same semantic type

    :param lexeme_offsets: A list of lexemes found in the sentence. Format should be a tuple of the lexeme itself,
     its character offset (can be either within sentence or document, as long as it is consistent),
     the concept code to which it corresponds, and a listing of its semantic types in a semicolon-delimited string
    :return: A list of concept code, lexeme, begin, end tuples consisting of only the longest distinct (non-overlapping)
     lexemes
    """
    # Convert lexeme_offsets into length, start, end, lexeme tuples and sort by increasing length.
    offset_tuples: List[Tuple[int, int, int, str, str, List[str]]] = []
    max_len = 0
    for lexeme_offset in lexeme_offsets:
        length: int = len(str(lexeme_offset[0]))
        start: int = int(str(lexeme_offset[1]))  # handle cases where input could be either a string or int
        semtypes_raw: str = str(lexeme_offset[3])
        semtypes: List[str] = []
        for semtype_str in semtypes_raw.split(';'):
            semtype = semtype_str.split(':')[0]
            semtypes.append(semtype)
        end: int = start + length
        max_len = max(max_len, end)
        offset_tuples.append((length, start, end, str(lexeme_offset[0]), str(lexeme_offset[2]), list(set(semtypes))))
    offset_tuples.sort(key=lambda t: t[0], reverse=False)
    # Now iterate through the list in descending order and populate already visited indices. If a subsequent
    # offset index is already populated, then that lexeme is subsumed and should be excluded
    sentence_occupied = [set([])] * max_len
    output_offsets = []
    for offset in offset_tuples:
        begin: int = offset[1]
        end: int = offset[2]
        lexeme: str = offset[3]
        concept_code: str = offset[4]
        semtypes: List[str] = offset[5]

        no_write_types = set()
        for i in range(begin, end):
            for semtype in sentence_occupied[i]:
                no_write_types.add(semtype)
                break
        write: bool = False
        for semtype in semtypes:
            if semtype not in no_write_types:
                for i in range(begin, end):
                    sentence_occupied[i].add(semtype)
                write = True
        if write:
            output_offsets.append((concept_code, lexeme, begin, end))
    return output_offsets


def flatten_overlaps_schema(concept: str, lexeme: str, begin: str, end: str) -> DataType:
    """:return: The schema returned by flatten_overlaps."""
    return ArrayType(StructType([
        StructField(concept, StringType(), False),
        StructField(lexeme, StringType(), False),
        StructField(begin, IntegerType(), False),
        StructField(end, IntegerType(), False)
    ]))
