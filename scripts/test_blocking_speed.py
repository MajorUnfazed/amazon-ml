"""Test fast inverted index blocking on 10,000 real entities."""

import os
import time
import re
from collections import defaultdict

train_dir = os.path.join("dataset", "train")

# Load 10,000 ground truth entries for US
gt_us = {}
s1_ids = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_us[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip())
            s1_ids.append(p[0])
            if len(s1_ids) >= 10000:
                break

needed_s1 = set(s1_ids)
target_partners = {pid for plist in gt_us.values() for pid in plist}

# Load S1 records
s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

# Load 200,000 partner records (including all target partners + noise)
partner_records = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in target_partners or len(partner_records) < 200000:
                partner_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

print(f"Loaded {len(s1_records):,} S1 queries and {len(partner_records):,} partner records.")
print(f"Target ground truth links to find: {len(target_partners):,}")

RE_WORD = re.compile(r"[a-z0-9]+")
RE_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9\-]+)\.(?:com|org|net|in|co\.in|fr|io|info|biz|co)\b", re.IGNORECASE)

STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "by", "from", "on", "to", "with",
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company",
    "services", "solutions", "enterprises", "group", "holdings", "management", "consulting", "international",
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd", "lane", "ln", "drive", "dr", "suite", "ste"
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

t0 = time.time()
print("\nBuilding multi-signal inverted index (tokens + core char-3grams + postal)...")
token_index = defaultdict(list)
char3_index = defaultdict(list)
postal_index = defaultdict(list)

for pid, prec in partner_records.items():
    toks = clean_tokens(prec["name"])
    addr_toks = clean_tokens(prec["addr"])[:3]
    for t in set(toks + addr_toks):
        token_index[t].append(pid)

    c3 = get_char_ngrams(prec["name"])
    for gram in set(c3):
        char3_index[gram].append(pid)

    post = extract_postal(prec["addr"])
    if post:
        postal_index[post].append(pid)

t_idx = time.time() - t0
print(f"Index built in {t_idx:.2f}s.")

# Query retrieval with multi-signal union
t0 = time.time()
hits = 0
total_gt_links = sum(len(plist) for plist in gt_us.values())
total_cands = 0

for s1_id, prec in s1_records.items():
    true_matches = gt_us.get(s1_id, set())
    toks = clean_tokens(prec["name"])
    addr_toks = clean_tokens(prec["addr"])[:3]

    cand_scores = defaultdict(float)

    # 1. Word token overlap (weight = 2.0)
    for t in set(toks + addr_toks):
        pids = token_index.get(t, [])
        if len(pids) <= 300:
            w = 3.0 if t in toks else 1.0
            for pid in pids:
                cand_scores[pid] += w

    # 2. Char 3-gram overlap (weight = 0.2)
    c3 = get_char_ngrams(prec["name"])
    for gram in set(c3):
        pids = char3_index.get(gram, [])
        if len(pids) <= 500:
            for pid in pids:
                cand_scores[pid] += 0.2

    # 3. Exact postal code match (boost if shared)
    post = extract_postal(prec["addr"])
    if post and post in postal_index:
        pids = postal_index[post]
        if len(pids) <= 200:
            for pid in pids:
                cand_scores[pid] += 2.5

    # Keep top 35 candidates
    if cand_scores:
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:35]
        cands = {pid for pid, _ in top_cands}
    else:
        cands = set()

    total_cands += len(cands)
    hits += len(true_matches & cands)

t_query = time.time() - t0
recall = hits / total_gt_links
avg_cands = total_cands / len(s1_records)

print(f"\nMulti-Signal Results on 10,000 S1 queries:")
print(f"Query Retrieval Time: {t_query:.2f}s ({len(s1_records)/t_query:.0f} queries/sec)")
print(f"Recall:               {recall:.2%} ({hits:,} / {total_gt_links:,} true links found)")
print(f"Avg Candidates / S1:  {avg_cands:.1f}")


