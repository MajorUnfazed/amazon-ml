"""Pure-function text and address normalization engine for LinkSure."""

import re
import unicodedata
from typing import Tuple, List, Optional, Dict, Any
from .rules import (
    ALL_LEGAL_SUFFIXES,
    LEGAL_PREFIXES,
    ADDRESS_ABBREVIATIONS,
    DBA_PATTERNS,
    LANDMARK_TRIGGERS
)

# Compile regex patterns once for maximum performance
RE_ACCENT = re.compile(r"[\u0300-\u036f]")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_AMP = re.compile(r"\s*&\s*")
RE_NUMBERS = re.compile(r"\b\d+[a-z]?\b")

# Postal code patterns
RE_POSTAL_IN = re.compile(r"\b[1-9][0-9]{5}\b")
RE_POSTAL_US = re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b")
RE_POSTAL_FR = re.compile(r"\b(?:0[1-9]|[1-8][0-9]|9[0-8])[0-9]{3}\b")

# Compile abbreviation patterns
COMPILED_ABBRS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBREVIATIONS.items()]
COMPILED_DBA = [re.compile(pattern, re.IGNORECASE) for pattern in DBA_PATTERNS]

# Regex for dotted acronyms like s.a.r.l. -> sarl, l.l.c. -> llc
RE_DOTTED_ABBR = re.compile(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z](?:\.|\b))")

def strip_accents(text: str) -> str:
    """Strip combining diacritical marks using standard library unicodedata."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

try:
    from unidecode import unidecode
except ImportError:
    unidecode = lambda x: x

def clean_string(text: str) -> str:
    """Lowercase, romanize non-Latin scripts (Indic/Tamil/Hindi/etc.), strip accents, collapse dotted acronyms, replace '&' with 'and', remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unidecode(str(text))
    text = strip_accents(text).lower()
    text = RE_DOTTED_ABBR.sub("", text)
    text = RE_AMP.sub(" and ", text)
    text = RE_PUNCT.sub(" ", text)
    text = RE_SPACES.sub(" ", text)
    return text.strip()

def extract_legal_suffix(name_clean: str) -> Tuple[str, str]:
    """
    Identifies and strips legal business suffixes AND prefixes (e.g. SARL, SAS, SCI, M/S).
    Returns (core_name, affix).
    """
    if not name_clean:
        return "", ""

    tokens = name_clean.split()
    if not tokens:
        return "", ""

    core = name_clean
    matched_affix = ""

    # 1. Strip prefix if present (e.g. "sarl boulangerie dupont" -> "boulangerie dupont")
    name_str = " " + core + " "
    for prefix in LEGAL_PREFIXES:
        pattern = r"^\s*" + re.escape(prefix) + r"\s+"
        if re.search(pattern, name_str):
            sub = re.sub(pattern, "", name_str).strip()
            if len(sub) >= 2:
                core = sub
                matched_affix = prefix
                break

    # 2. Strip suffix if present (e.g. "dupont boulangerie sarl" -> "dupont boulangerie")
    name_str = " " + core + " "
    for suffix in ALL_LEGAL_SUFFIXES:
        pattern = r"\s+" + re.escape(suffix) + r"\s*$"
        if re.search(pattern, name_str):
            sub = re.sub(pattern, "", name_str).strip()
            if len(sub) >= 2:
                core = sub
                if not matched_affix:
                    matched_affix = suffix
                break

    return core, matched_affix

def extract_dba_variants(raw_name: str) -> List[str]:
    """
    Detects Doing-Business-As patterns and splits into name variants.
    """
    variants = []
    cleaned = clean_string(raw_name)
    if not cleaned:
        return [""]

    split_done = False
    for pattern in COMPILED_DBA:
        if pattern.search(cleaned):
            parts = pattern.split(cleaned)
            for p in parts:
                p_clean = p.strip()
                if len(p_clean) >= 2:
                    variants.append(p_clean)
            split_done = True
            break

    if not split_done:
        variants.append(cleaned)

    return list(dict.fromkeys(variants))

def extract_acronym(core_name: str) -> str:
    """Extracts initials from multi-word business core names."""
    tokens = [t for t in core_name.split() if len(t) > 0 and t not in {"and", "the", "of", "for"}]
    if len(tokens) >= 2:
        return "".join(t[0] for t in tokens)
    return ""

def extract_postal_code(address: str, country: Optional[str] = None) -> str:
    """
    Extracts 5/6 digit postal or PIN codes based on country context.
    """
    if not address:
        return ""

    c_upper = country.upper() if country else ""

    if c_upper == "INDIA" or c_upper == "IN":
        m = RE_POSTAL_IN.search(address)
        if m:
            return m.group(0)
    elif c_upper == "US" or c_upper == "USA":
        m = RE_POSTAL_US.search(address)
        if m:
            return m.group(0)[:5]
    elif c_upper == "FRANCE" or c_upper == "FR":
        m = RE_POSTAL_FR.search(address)
        if m:
            return m.group(0)

    # Country agnostic fallback search
    m_in = RE_POSTAL_IN.search(address)
    if m_in:
        return m_in.group(0)

    m_fr = RE_POSTAL_FR.search(address)
    if m_fr:
        return m_fr.group(0)

    m_us = RE_POSTAL_US.search(address)
    if m_us:
        return m_us.group(0)[:5]

    return ""

# Direct dictionary mapping for fast token-based address abbreviations
FAST_ABBR_MAP = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "apt": "apartment", "ste": "suite", "bldg": "building", "fl": "floor",
    "ctr": "center", "sq": "square", "terr": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mkt": "market", "opp": "opposite", "nr": "near", "extn": "extension", "ext": "extension",
    "col": "colony", "soc": "society",
    "salai": "road", "marg": "road", "rasta": "road", "gali": "lane",
    "chowk": "square", "bazar": "market", "bazaar": "market",
    "bd": "boulevard", "r": "rue", "chem": "chemin", "rte": "route", "all": "allee", "faub": "faubourg",
    "cours": "cours", "quai": "quai", "pass": "passage"
}

RE_PHONE = re.compile(r"\b(?:ph|tel|mob|phone)?[:.\s]*([6-9]\d{9}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b", re.IGNORECASE)

def extract_phone_numbers(address: str) -> List[str]:
    """Extracts 10-digit mobile/landline numbers from address strings."""
    if not address:
        return []
    matches = RE_PHONE.findall(str(address))
    return [re.sub(r"\D", "", m) for m in matches if len(re.sub(r"\D", "", m)) >= 10]

def clean_address(address: str) -> str:
    """Cleans address string and expands street and locality abbreviations."""
    cleaned = clean_string(address)
    if not cleaned:
        return ""

    tokens = cleaned.split()
    expanded = [FAST_ABBR_MAP.get(t, t) for t in tokens]
    return " ".join(expanded)

def extract_building_numbers(address: str) -> List[str]:
    """Extracts house/building/unit numbers."""
    if not address:
        return []
    cleaned = clean_string(address)
    numbers = RE_NUMBERS.findall(cleaned)
    # Filter out 5/6 digit postal codes from building numbers
    return [n for n in numbers if len(n) <= 4]

def extract_landmarks(address: str) -> List[str]:
    """Finds phrases following landmark cues."""
    if not address:
        return []
    cleaned = clean_string(address)
    tokens = cleaned.split()
    landmarks = []

    for i, token in enumerate(tokens):
        if token in LANDMARK_TRIGGERS:
            # Grab following 2-3 words
            phrase = " ".join(tokens[i+1:i+4])
            if phrase:
                landmarks.append(phrase)

    return landmarks

def normalize_record(name: str, address: str, country: str) -> Dict[str, Any]:
    """
    Takes raw entity attributes and returns rich normalized feature dictionary.
    """
    name_clean = clean_string(name)
    core_name, suffix = extract_legal_suffix(name_clean)
    dba_variants = extract_dba_variants(name)
    acronym = extract_acronym(core_name)

    country_clean = clean_string(country)
    address_clean = clean_address(address)
    postal_code = extract_postal_code(address, country)
    building_numbers = extract_building_numbers(address)
    landmarks = extract_landmarks(address)

    return {
        "name_clean": name_clean,
        "name_core": core_name,
        "legal_suffix": suffix,
        "name_variants": dba_variants,
        "name_acronym": acronym,
        "country_clean": country_clean,
        "address_clean": address_clean,
        "postal_code": postal_code,
        "building_numbers": building_numbers,
        "landmarks": landmarks,
        "phone_numbers": extract_phone_numbers(address)
    }
