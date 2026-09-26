"""Dictionary tables and regex definitions for LinkSure normalization."""

import re

# Legal Suffixes by Country/Language
US_LEGAL_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp",
    "limited liability company", "llc", "l.l.c.",
    "company", "co", "limited", "ltd", "l.t.d."
]

INDIA_LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "p ltd", "p. ltd.",
    "limited", "ltd", "l.t.d.",
    "limited liability partnership", "llp", "l.l.p.",
    "proprietorship", "enterprises", "enterprise",
    "traders", "trading company", "agency", "agencies", "associates"
]

FRANCE_LEGAL_SUFFIXES = [
    "societe a responsabilite limitee", "sarl", "s.a.r.l.",
    "societe par actions simplifiee unipersonnelle", "sasu", "s.a.s.u.",
    "societe par actions simplifiee", "sas", "s.a.s.",
    "societe anonyme", "sa", "s.a.",
    "entreprise unipersonnelle a responsabilite limitee", "eurl", "e.u.r.l.",
    "societe en nom collectif", "snc", "s.n.c.",
    "societe civile immobiliere", "sci", "s.c.i.",
    "societe en commandite simple", "scs",
    "societe d'exercice liberal a responsabilite limitee", "selarl",
    "et cie", "et compagnie", "cie"
]

# Combined unique suffixes sorted by length descending for greedy replacement
ALL_LEGAL_SUFFIXES = sorted(
    list(set(US_LEGAL_SUFFIXES + INDIA_LEGAL_SUFFIXES + FRANCE_LEGAL_SUFFIXES)),
    key=lambda s: len(s),
    reverse=True
)

# Legal Prefixes (common in France: SARL, SAS, SCI, and India: M/S)
LEGAL_PREFIXES = sorted([
    "societe a responsabilite limitee",
    "societe par actions simplifiee unipersonnelle",
    "societe par actions simplifiee",
    "societe anonyme",
    "entreprise unipersonnelle a responsabilite limitee",
    "societe civile immobiliere",
    "societe en nom collectif",
    "societe d exercice liberal",
    "sarl", "sas", "sasu", "eurl", "sci", "snc", "sa", "selarl", "scs", "sca",
    "ste", "societe", "ets", "etablissements", "cie", "compagnie",
    "m s", "ms", "messrs"
], key=lambda s: len(s), reverse=True)

# Address Abbreviations
ADDRESS_ABBREVIATIONS = {
    # English / US
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
    r"\bctr\b": "center",
    r"\bsq\b": "square",
    r"\bterr\b": "terrace",
    # India
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\badj\b": "adjacent",
    r"\bb/h\b": "behind",
    r"\bbh\b": "behind",
    r"\bext\b": "extension",
    r"\bmkt\b": "market",
    # France
    r"\br\.\b": "rue",
    r"\br\b": "rue",
    r"\bbd\.\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bchem\.\b": "chemin",
    r"\bchem\b": "chemin",
    r"\ball\.\b": "allee",
    r"\bimp\.\b": "impasse",
    r"\bimp\b": "impasse",
    r"\brte\b": "route",
    r"\bfaub\b": "faubourg",
    r"\bz\.?i\.?\b": "zone industrielle",
    r"\bz\.?a\.?\b": "zone activite"
}

# DBA Markers
DBA_PATTERNS = [
    r"\bd/b/a\b",
    r"\bdba\b",
    r"\bt/a\b",
    r"\btrading as\b",
    r"\ba/k/a\b",
    r"\baka\b",
    r"\bdoing business as\b"
]

# Landmark Trigger Words
LANDMARK_TRIGGERS = [
    "near", "opposite", "opp", "behind", "next to", "beside", "adjacent to",
    "close to", "in front of", "pres de", "a cote de", "en face de", "face a"
]
