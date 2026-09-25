"""Inspect the missed pairs to see what signals link them."""

import os
import re
from collections import defaultdict
from rapidfuzz import fuzz

train_dir = os.path.join("dataset", "train")

# Load 5,000 ground truth entries for US
gt_us = {}
s1_ids = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_us[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip())
            s1_ids.append(p[0])
            if len(s1_ids) >= 5000:
                break

needed_s1 = set(s1_ids)
target_partners = {pid for plist in gt_us.values() for pid in plist}

s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

partner_records = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in target_partners or len(partner_records) < 150000:
                partner_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

RE_WORD = re.compile(r"[a-z0-9]+")
RE_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9\-]+)\.(?:com|org|net|in|co\.in|fr|io|info|biz|co)\b", re.IGNORECASE)
STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "by", "from", "on", "to", "with",
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company"
}

def clean_tokens(text: str):
    m = RE_DOMAIN.search(text)
    if m:
        text = text + " " + m.group(1)
    toks = RE_WORD.findall(text.lower())
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS]

def get_char_ngrams(s: str, n=3):
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

RE_POSTAL = re.compile(r"\b\d{5,6}\b")
def extract_postal(addr: str):
    m = RE_POSTAL.findall(addr)
    return m[0] if m else None

token_index = defaultdict(list)
char3_index = defaultdict(list)
postal_index = defaultdict(list)

for pid, prec in partner_records.items():
    toks = clean_tokens(prec["name"])
    addr_toks = clean_tokens(prec["addr"])[:3]
    for t in set(toks + addr_toks):
        token_index[t].append(pid)
    for gram in set(get_char_ngrams(prec["name"])):
        char3_index[gram].append(pid)
    post = extract_postal(prec["addr"])
    if post:
        postal_index[post].append(pid)

missed = []
for s1_id, prec in s1_records.items():
    true_matches = gt_us.get(s1_id, set())
    toks = clean_tokens(prec["name"])
    addr_toks = clean_tokens(prec["addr"])[:3]
    cand_scores = defaultdict(float)

    for t in set(toks + addr_toks):
        for pid in token_index.get(t, [])[:300]:
            cand_scores[pid] += 3.0 if t in toks else 1.0

    for gram in set(get_char_ngrams(prec["name"])):
        for pid in char3_index.get(gram, [])[:500]:
            cand_scores[pid] += 0.2

    post = extract_postal(prec["addr"])
    if post and post in postal_index:
        for pid in postal_index[post][:200]:
            cand_scores[pid] += 2.5

    top_cands = {pid for pid, _ in sorted(cand_scores.items(), key=lambda x: -x[1])[:40]}
    for pid in true_matches:
        if pid not in top_cands and pid in partner_records:
            missed.append((s1_id, pid))

print(f"Total Missed Pairs: {len(missed):,}")
print("Examining 15 sample missed pairs:\n")
for s1_id, pid in missed[:15]:
    s1 = s1_records[s1_id]
    p = partner_records[pid]
    print(f"S1 [{s1_id}]: {s1['name']} | {s1['addr']}")
    print(f"P  [{pid}]:  {p['name']} | {p['addr']}")
    print(f"Name fuzz: {fuzz.token_sort_ratio(s1['name'], p['name'])}% | Addr fuzz: {fuzz.token_set_ratio(s1['addr'], p['addr'])}%")
    print("-" * 60)
