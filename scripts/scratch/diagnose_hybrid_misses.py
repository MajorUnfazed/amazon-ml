import os
import sys
import time
import math
import heapq
import random
from collections import defaultdict
import pandas as pd

src_dir = os.path.join(os.getcwd(), "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from normalize.text import normalize_record

# 1. Load 500 US S1 queries from train
N_QUERIES = 500
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=2000, dtype=str).fillna('')
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

# 2. Load partners
partner_lookup = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4 and p[3] == 'US':
                pid = p[0]
                if pid in all_true_partners or (len(partner_lookup) < 100000 and random.random() < 0.05):
                    norm = normalize_record(p[1], p[2], p[3])
                    norm['entity_id'] = pid
                    norm['country'] = 'US'
                    partner_lookup[pid] = norm

# 3. Build index
slug_index = defaultdict(list)
addr_key_index = defaultdict(list)
word_index = defaultdict(list)
char3_index = defaultdict(list)

for pid, norm in partner_lookup.items():
    n_clean = str(norm["name_clean"])
    n_core = str(norm["name_core"])
    slug = n_core.replace(" ", "")[:25]
    if len(slug) >= 4:
        slug_index[slug].append(pid)
    nums = norm.get("building_numbers", [])
    a_toks = [t for t in str(norm["address_clean"]).split() if len(t) >= 3 and not t.isdigit()]
    if nums and a_toks:
        addr_k = f"{nums[0]}_{a_toks[0]}"
        addr_key_index[addr_k].append(pid)
    for w in set(n_clean.split()):
        if len(w) >= 3:
            word_index[w].append(pid)
    core_flat = n_core.replace(" ", "")
    c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
    for c3 in c3_set:
        char3_index[c3].append(pid)

# Check which true partners are retrieved at ALL (union of all posting lists, without top-K cut)
unbounded_hits = 0
total_gt = sum(len(v) for v in eval_s1_dict.values())
missed_at_all = []

for eid, name, addr, c in s1_records:
    norm = normalize_record(name, addr, c)
    true_set = eval_s1_dict[eid]
    cands = set()
    
    # Slug
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        cands.update(slug_index.get(slug, []))
        
    # Address key
    nums = norm.get("building_numbers", [])
    a_toks = [t for t in str(norm["address_clean"]).split() if len(t) >= 3 and not t.isdigit()]
    if nums and a_toks:
        addr_k = f"{nums[0]}_{a_toks[0]}"
        cands.update(addr_key_index.get(addr_k, []))
        
    # Words
    for w in set(str(norm["name_clean"]).split()):
        if len(w) >= 3:
            cands.update(word_index.get(w, []))
            
    # Char 3-grams
    core_flat = str(norm["name_core"]).replace(" ", "")
    c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
    for c3 in c3_set:
        cands.update(char3_index.get(c3, []))
        
    hits = cands.intersection(true_set)
    unbounded_hits += len(hits)
    for m in true_set - hits:
        if m in partner_lookup:
            missed_at_all.append((eid, name, addr, m, partner_lookup[m]))

print(f"UNBOUNDED Inverted Index Candidate Recall: {unbounded_hits} / {total_gt} ({unbounded_hits / total_gt:.2%})")
print(f"Total true partners completely missed: {len(missed_at_all)}")
for eid, name, addr, m, p_norm in missed_at_all[:10]:
    print(f"S1 ({eid}): {name}  ||  {addr}")
    print(f"P  ({m}): {p_norm['name_clean']}  ||  {p_norm['address_clean']}")
    print("-" * 60)
