import os
import sys
import time
import math
import heapq
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import joblib

src_dir = os.path.join(os.getcwd(), "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from normalize.text import normalize_record
from features.builder import FeatureBuilder
from decide.one_to_one import resolve_one_to_one

print("=" * 70)
print("TESTING CANDIDATE RETRIEVAL & MODEL SCORING ON CONTROLLED GT SET")
print("=" * 70)

# 1. Select 1,000 S1 queries from train (US)
N_QUERIES = 1000
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=5000, dtype=str).fillna('')
gt_dict = {row['source1_entity_id']: set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set() for _, row in gt_df.iterrows()}

s1_records = []
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8', errors='ignore') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt_dict and p[3] == 'US':
            s1_records.append((p[0], p[1], p[2], p[3]))
            if len(s1_records) >= N_QUERIES:
                break

eval_s1_dict = {r[0]: gt_dict[r[0]] for r in s1_records}
all_true_partners = {pid for v in eval_s1_dict.values() for pid in v}
total_true_matches = sum(len(v) for v in eval_s1_dict.values())
print(f"Loaded {len(s1_records)} US S1 queries with {total_true_matches} true partner matches ({len(all_true_partners)} unique partner IDs).")

# 2. Load ALL true partners + 150,000 background partners
partner_lookup = {}
t0 = time.time()
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4 and p[3] == 'US':
                pid = p[0]
                if pid in all_true_partners:
                    norm = normalize_record(p[1], p[2], p[3])
                    norm['entity_id'] = pid
                    norm['country'] = 'US'
                    partner_lookup[pid] = norm
                elif len(partner_lookup) < 150000 and random.random() < 0.05:
                    norm = normalize_record(p[1], p[2], p[3])
                    norm['entity_id'] = pid
                    norm['country'] = 'US'
                    partner_lookup[pid] = norm

found_true = sum(1 for pid in all_true_partners if pid in partner_lookup)
print(f"Partner pool size: {len(partner_lookup):,} records.")
print(f"True partners in pool: {found_true} / {len(all_true_partners)} ({found_true/len(all_true_partners):.2%}) in {time.time() - t0:.1f}s.")

# 3. Test OLD retriever from generate_test_predictions.py
print("\n--- Testing OLD Retriever (generate_test_predictions.py) ---")
token_index_old = defaultdict(list)
slug_index_old = defaultdict(list)

for pid, norm in partner_lookup.items():
    toks = set(str(norm["name_clean"]).split() + str(norm["address_clean"]).split()[:3])
    for t in toks:
        if len(t) >= 2:
            token_index_old[t].append(pid)
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        slug_index_old[slug].append(pid)

hits_old = 0
for eid, name, addr, c in s1_records:
    norm = normalize_record(name, addr, c)
    true_set = eval_s1_dict[eid]
    cand_scores = defaultdict(float)

    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        for pid in slug_index_old.get(slug, []):
            cand_scores[pid] += 8.0

    toks = set(str(norm["name_clean"]).split() + str(norm["address_clean"]).split()[:3])
    for t in toks:
        pids = token_index_old.get(t, [])
        if 0 < len(pids) <= 300:
            w = 3.0 if t in str(norm["name_clean"]) else 1.0
            for pid in pids:
                cand_scores[pid] += w

    if cand_scores:
        top_cands = set(x[0] for x in heapq.nlargest(25, cand_scores.items(), key=lambda x: x[1]))
    else:
        top_cands = set()
    hits_old += len(top_cands.intersection(true_set))

print(f"OLD Retriever Recall (K=25): {hits_old} / {total_true_matches} ({hits_old / total_true_matches:.2%})")

# 4. Now let's diagnose why the other matches were missed
missed_examples = []
for eid, name, addr, c in s1_records[:100]:
    norm = normalize_record(name, addr, c)
    true_set = eval_s1_dict[eid]
    cand_scores = defaultdict(float)
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        for pid in slug_index_old.get(slug, []):
            cand_scores[pid] += 8.0
    toks = set(str(norm["name_clean"]).split() + str(norm["address_clean"]).split()[:3])
    for t in toks:
        pids = token_index_old.get(t, [])
        if 0 < len(pids) <= 300:
            w = 3.0 if t in str(norm["name_clean"]) else 1.0
            for pid in pids:
                cand_scores[pid] += w
    top_cands = set(x[0] for x in heapq.nlargest(25, cand_scores.items(), key=lambda x: x[1])) if cand_scores else set()
    missed = true_set - top_cands
    for m_pid in missed:
        if m_pid in partner_lookup:
            missed_examples.append((eid, name, addr, m_pid, partner_lookup[m_pid]))

print(f"\nSample of missed matches ({len(missed_examples)} total missed in first 100 S1):")
for eid, s1_name, s1_addr, m_pid, p_norm in missed_examples[:5]:
    print(f"S1 ({eid}): {s1_name}  ||  {s1_addr}")
    print(f"P  ({m_pid}): {p_norm['name_clean']}  ||  {p_norm['address_clean']}")
    print("-" * 60)
