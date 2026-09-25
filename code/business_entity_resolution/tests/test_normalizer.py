"""Unit tests for LinkSure Normalization Engine."""

import os
import sys

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalize.text import (
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

def test_accent_stripping():
    assert strip_accents("Café Résumé") == "Cafe Resume"
    assert strip_accents("München Über") == "Munchen Uber"
    assert strip_accents("Hôtel de Ville") == "Hotel de Ville"

def test_clean_string():
    assert clean_string("Ben & Jerry's, Inc.") == "ben and jerry s inc"
    assert clean_string("  Spaces   &   Tabs\t") == "spaces and tabs"

def test_legal_suffixes():
    # US
    core, suffix = extract_legal_suffix("apple inc")
    assert core == "apple"
    assert suffix == "inc"

    core, suffix = extract_legal_suffix("widgets llc")
    assert core == "widgets"
    assert suffix == "llc"

    # India
    core, suffix = extract_legal_suffix("reliance retail pvt ltd")
    assert core == "reliance retail"
    assert suffix in {"pvt ltd", "private limited"}

    # France
    core, suffix = extract_legal_suffix("boulangerie dubois sarl")
    assert core == "boulangerie dubois"
    assert suffix == "sarl"

    core, suffix = extract_legal_suffix("renault sas")
    assert core == "renault"
    assert suffix == "sas"

def test_dba_detection():
    variants = extract_dba_variants("Acme Holdings dba Super Rocket Store")
    assert len(variants) == 2
    assert variants[0] == "acme holdings"
    assert variants[1] == "super rocket store"

def test_postal_code_extraction():
    # India PIN
    assert extract_postal_code("123 MG Road, Bengaluru, 560001", "India") == "560001"
    # US ZIP
    assert extract_postal_code("100 Main St, New York, NY 10001-4321", "US") == "10001"
    # France Postal Code
    assert extract_postal_code("15 Rue de Rivoli, 75004 Paris", "France") == "75004"

def test_address_abbreviations():
    addr = clean_address("12 Av. des Champs-Élysées, 75008 Paris")
    assert "avenue" in addr
    assert "champs elysees" in addr

    addr_in = clean_address("Shop 4, Opp City Hospital, Nr Clock Tower")
    assert "opposite" in addr_in
    assert "near" in addr_in

def test_landmarks():
    lms = extract_landmarks("Opposite Grand Hotel, Near Railway Station")
    assert len(lms) >= 1

def test_normalize_record():
    rec = normalize_record(
        name="L'Étoile du Nord S.A.R.L.",
        address="42 Bd. Saint-Germain, 75005 Paris",
        country="France"
    )
    assert "etoile du nord" in rec["name_core"]
    assert rec["legal_suffix"] in {"sarl", "s.a.r.l."}
    assert "boulevard" in rec["address_clean"]
    assert rec["postal_code"] == "75005"

if __name__ == "__main__":
    test_accent_stripping()
    test_clean_string()
    test_legal_suffixes()
    test_dba_detection()
    test_postal_code_extraction()
    test_address_abbreviations()
    test_landmarks()
    test_normalize_record()
    print("All normalizer tests passed successfully!")
