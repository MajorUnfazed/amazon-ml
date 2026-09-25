"""Evaluate multi-retriever recall on 5,000 real GT entities."""

import os
import random
from collections import Counter
from rapidfuzz import fuzz

random.seed(42)
train_dir = os.path.join("dataset", "train")

# Load 5,000 random non-singleton S1 entities from GT
gt_map = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_map[p[0]] = [x.strip() for x in p[1].split(",") if x.strip()]

sample_s1_keys = set(random.sample(list(gt_map.keys()), 3000))
sample_gt = {k: gt_map[k] for k in sample_s1_keys}
sample_partner_ids = {pid for plist in sample_gt.values() for pid in plist}

# Load S1 records
sample_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in sample_s1_keys:
            sample_s1_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

# Load partner records
sample_partner_records = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in sample_partner_ids:
                sample_partner_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

print(f"Sampled {len(sample_s1_records):,} S1 entities with {len(sample_partner_records):,} true partner links.")

# Analyze similarities between True Matches
name_fuzz_scores = []
addr_fuzz_scores = []
either_high = 0
both_high = 0
total_links = 0

for s1_id, partners in sample_gt.items():
    s1 = sample_s1_records.get(s1_id)
    if not s1:
        continue
    for pid in partners:
        p = sample_partner_records.get(pid)
        if not p:
            continue
        total_links += 1

        n_score = fuzz.token_sort_ratio(s1["name"].lower(), p["name"].lower())
        a_score = fuzz.token_set_ratio(s1["addr"].lower(), p["addr"].lower()) if (s1["addr"] and p["addr"]) else 0

        name_fuzz_scores.append(n_score)
        addr_fuzz_scores.append(a_score)

        if n_score >= 70 or a_score >= 70:
            either_high += 1
        if n_score >= 70 and a_score >= 70:
            both_high += 1

print(f"Total True Links Analyzed: {total_links:,}")
print(f"Mean Name Token Sort Ratio:   {sum(name_fuzz_scores)/len(name_fuzz_scores):.1f}%")
print(f"Mean Address Token Set Ratio: {sum(addr_fuzz_scores)/len(addr_fuzz_scores):.1f}%")
print(f"Links with Name >= 70% OR Addr >= 70%: {either_high:,} ({either_high/total_links:.2%})")
print(f"Links with BOTH Name >= 70% AND Addr >= 70%: {both_high:,} ({both_high/total_links:.2%})")
