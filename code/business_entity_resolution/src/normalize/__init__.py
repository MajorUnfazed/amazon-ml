"""Normalization module for LinkSure."""
from .text import (
    strip_accents,
    clean_string,
    clean_address,
    extract_legal_suffix,
    extract_dba_variants,
    extract_acronym,
    extract_postal_code,
    extract_building_numbers,
    extract_landmarks,
    normalize_record
)

__all__ = [
    "strip_accents",
    "clean_string",
    "clean_address",
    "extract_legal_suffix",
    "extract_dba_variants",
    "extract_acronym",
    "extract_postal_code",
    "extract_building_numbers",
    "extract_landmarks",
    "normalize_record"
]
