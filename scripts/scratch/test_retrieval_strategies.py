import os
import sys
import time
import math
import heapq
from collections import defaultdict
import pandas as pd
import numpy as np

src_dir = os.path.join(os.getcwd(), "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from normalize.text import normalize_record

print("=" * 70)
print("BENCHMARKING HIGH-RECALL CANDIDATE GENERATION STRATEGIES")
print("=" * 70)

# 1. Load 2,000 S1 queries from train (US)
N_QUERIES = 2000
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000, dtype=str).fillna('')
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
total_true_matches = sum(len(v) for v in eval_s1_dict.values())
true_partners = {pid for v in eval_s1_dict.values() for pid in v}
print(f"Loaded {len(s1_records)} US S1 queries with {total_true_matches} true partner matches.")

# 2. Load 500,000 US partner records
partner_lookup = {}
t0 = time.time()
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4 and p[3] == 'US':
                pid = p[0]
                # Keep true partners + up to 500,000 background partners
                if pid in true_partners or len(partner_lookup) < 500000:
                    norm = normalize_record(p[1], p[2], p[3])
                    norm['entity_id'] = pid
                    partner_lookup[pid] = norm
            if len(partner_lookup) >= 500000:
                break

print(f"Loaded {len(partner_lookup):,} partner records in {time.time() - t0:.1f}s.")

# 3. Test different indexing approaches
# Approach A: Multi-channel inverted index with IDF weighting and address key
print("\nBuilding multi-channel index with IDF...")
t0 = time.time()

# Frequency count
token_df = defaultdict(int)
for pid, norm in partner_lookup.items():
    toks = set(str(norm['name_clean']).split())
    for t in toks:
        if len(t) >= 3:
            token_df[t] += 1

N_docs = len(partner_lookup)
token_idf = {t: math.log(1.0 + N_docs / df) for t, df in token_df.items() if df <= 50000}

# Build posting lists
slug_index = defaultdict(list)
token_index = defaultdict(list)
addr_key_index = defaultdict(list)
char3_index = defaultdict(list)

for pid, norm in partner_lookup.items():
    # 1. Exact core slug
    slug = str(norm['name_core']).replace(' ', '')[:25]
    if len(slug) >= 4:
        slug_index[slug].append(pid)
    
    # 2. Token index with IDF
    toks = set(str(norm['name_clean']).split())
    for t in toks:
        if t in token_idf:
            token_index[t].append(pid)
            
    # 3. Address key: number + street token
    nums = norm.get('address_numbers', [])
    a_toks = [t for t in str(norm['address_clean']).split() if len(t) >= 3 and not t.isdigit()]
    if nums and a_toks:
        addr_k = f"{nums[0]}_{a_toks[0]}"
        addr_key_index[addr_k].append(pid)

print(f"Index built in {time.time() - t0:.1f}s.")

# Evaluate candidate recall for K = 25
print("\nRunning candidate retrieval evaluation (K=25)...")
t0 = time.time()
hits = 0
for eid, name, addr, c in s1_records:
    norm = normalize_record(name, addr, c)
    true_set = eval_s1_dict[eid]
    cand_scores = defaultdict(float)
    
    # Channel 1: Exact slug (strongest signal)
    slug = str(norm['name_core']).replace(' ', '')[:25]
    if len(slug) >= 4:
        for pid in slug_index.get(slug, []):
            cand_scores[pid] += 20.0
            
    # Channel 2: Address key (handles different names at exact same address)
    nums = norm.get('address_numbers', [])
    a_toks = [t for t in str(norm['address_clean']).split() if len(t) >= 3 and not t.isdigit()]
    if nums and a_toks:
        addr_k = f"{nums[0]}_{a_toks[0]}"
        pids = addr_key_index.get(addr_k, [])
        if 0 < len(pids) <= 200:
            for pid in pids:
                cand_scores[pid] += 12.0
                
    # Channel 3: Token IDF matching
    toks = set(str(norm['name_clean']).split())
    for t in toks:
        idf = token_idf.get(t, 0.0)
        if idf > 2.0:  # Only meaningful tokens
            pids = token_index.get(t, [])
            if 0 < len(pids) <= 2000:
                w = idf
                for pid in pids:
                    cand_scores[pid] += w
                    
    if cand_scores:
        top_cands = set(x[0] for x in heapq.nlargest(25, cand_scores.items(), key=lambda x: x[1]))
    else:
        top_cands = set()
        
    hits += len(top_cands.intersection(true_set))

elapsed = time.time() - t0
print(f"Results for K=25 across 500k partner pool:")
print(f"  Total true matches: {total_true_matches}")
print(f"  Retrieved hits: {hits}")
print(f"  CANDIDATE RECALL: {hits / total_true_matches:.2%}")
print(f"  Retrieval speed: {len(s1_records) / elapsed:.1f} queries/sec ({elapsed:.2f}s total)")
print("=" * 70)
