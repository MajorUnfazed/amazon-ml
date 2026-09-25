"""Test improved multi-signal blocking with street-number keys, domain slugs, and IDF."""

import os
import re
import math
from collections import defaultdict

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
RE_NUM_STREET = re.compile(r"\b(\d{1,6})\s*[,-]?\s*([a-z]{3,})\b")
RE_POSTAL = re.compile(r"\b\d{5,6}\b")

def get_name_slug(name: str) -> str:
    m = RE_DOMAIN.search(name)
    if m:
        name = m.group(1)
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    return s[:30]

def get_addr_keys(addr: str):
    addr_l = addr.lower()
    keys = []
    # 1. Street number + street name token
    m = RE_NUM_STREET.search(addr_l)
    if m:
        num, st = m.group(1), m.group(2)
        if st not in {"street", "road", "avenue", "lane", "drive", "court", "suite", "floor"}:
            keys.append(f"{num}_{st}")
        else:
            keys.append(num)
    # 2. Postal code
    p = RE_POSTAL.search(addr_l)
    if p:
        keys.append(f"post_{p.group(0)}")
    return keys

def clean_tokens(text: str):
    m = RE_DOMAIN.search(text)
    if m:
        text = text + " " + m.group(1)
    toks = RE_WORD.findall(text.lower())
    return [t for t in toks if len(t) >= 2]

# Compute document frequency on partner pool
print("Computing partner token DF...")
df_counts = defaultdict(int)
for pid, prec in partner_records.items():
    toks = set(clean_tokens(prec["name"]) + clean_tokens(prec["addr"])[:3])
    for t in toks:
        df_counts[t] += 1

N_pool = len(partner_records)
idf = {t: math.log(1.0 + N_pool / (1.0 + c)) for t, c in df_counts.items()}

# Build inverted indexes
print("Building inverted indices...")
token_index = defaultdict(list)
slug_index = defaultdict(list)
addr_key_index = defaultdict(list)

for pid, prec in partner_records.items():
    # 1. Tokens
    toks = set(clean_tokens(prec["name"]) + clean_tokens(prec["addr"])[:3])
    for t in toks:
        if df_counts[t] <= 1500: # only cap extreme words like 'the'
            token_index[t].append(pid)

    # 2. Slug
    slug = get_name_slug(prec["name"])
    if len(slug) >= 4:
        slug_index[slug].append(pid)

    # 3. Addr keys
    for k in get_addr_keys(prec["addr"]):
        addr_key_index[k].append(pid)

print(f"Indices built. Testing retrieval on {len(s1_records):,} queries...")

hits = 0
total_gt_links = sum(len(plist) for plist in gt_us.values())
total_cands = 0

for s1_id, prec in s1_records.items():
    true_matches = gt_us.get(s1_id, set())
    cand_scores = defaultdict(float)

    # Signal 1: Name slug exact match (weight 10.0)
    s1_slug = get_name_slug(prec["name"])
    if len(s1_slug) >= 4:
        for pid in slug_index.get(s1_slug, []):
            cand_scores[pid] += 10.0

    # Signal 2: Address key (number + street or postal) (weight 5.0)
    for k in get_addr_keys(prec["addr"]):
        pids = addr_key_index.get(k, [])
        if len(pids) <= 200:
            for pid in pids:
                cand_scores[pid] += 5.0

    # Signal 3: Name and address tokens with IDF weighting
    s1_toks = clean_tokens(prec["name"])
    s1_addr_toks = clean_tokens(prec["addr"])[:3]

    for t in set(s1_toks):
        w = idf.get(t, 1.0) * 1.5
        for pid in token_index.get(t, [])[:300]:
            cand_scores[pid] += w

    for t in set(s1_addr_toks):
        w = idf.get(t, 1.0) * 0.8
        for pid in token_index.get(t, [])[:200]:
            cand_scores[pid] += w

    # Keep top 40 candidates
    if cand_scores:
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:40]
        cands = {pid for pid, _ in top_cands}
    else:
        cands = set()

    total_cands += len(cands)
    hits += len(true_matches & cands)

recall = hits / total_gt_links
avg_cands = total_cands / len(s1_records)

print(f"\nIMPROVED RETRIEVAL RESULTS:")
print(f"Recall:              {recall:.2%} ({hits:,} / {total_gt_links:,} true links found)")
print(f"Avg Candidates / S1: {avg_cands:.1f}")
