import os
import sys
import pandas as pd
import re

src_dir = os.path.join(os.getcwd(), "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from normalize.text import normalize_record

# Load GT
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=100, dtype=str).fillna('')
pairs = []
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    m = row['matched_entity_ids']
    if m:
        for p in m.split(','):
            pairs.append((s1, p))

s1_needed = {x[0] for x in pairs}
p_needed = {x[1] for x in pairs}

s1_lookup = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8', errors='ignore') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in s1_needed:
            s1_lookup[p[0]] = (p[1], p[2], p[3])
            if len(s1_lookup) == len(s1_needed):
                break

p_lookup = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in p_needed:
                p_lookup[p[0]] = (p[1], p[2], p[3])
                if len(p_lookup) == len(p_needed):
                    break

match_types = {
    'exact_name': 0,
    'high_name_sim': 0,
    'exact_addr': 0,
    'high_addr_sim': 0,
    'both': 0,
    'name_diff_addr_same': 0,
    'addr_empty': 0,
}

print(f"Loaded {len(s1_lookup)} S1, {len(p_lookup)} P records from ground truth.")
count = 0
for s1, p in pairs:
    if s1 in s1_lookup and p in p_lookup:
        count += 1
        s1_n, s1_a, s1_c = s1_lookup[s1]
        p_n, p_a, p_c = p_lookup[p]
        s1_norm = normalize_record(s1_n, s1_a, s1_c)
        p_norm = normalize_record(p_n, p_a, p_c)
        
        name_exact = (s1_norm['name_clean'] == p_norm['name_clean']) or (s1_norm['name_core'] == p_norm['name_core'])
        addr_exact = bool(s1_norm['address_clean'] and s1_norm['address_clean'] == p_norm['address_clean'])
        addr_empty = not bool(p_norm['address_clean']) or not bool(s1_norm['address_clean'])
        
        if name_exact:
            match_types['exact_name'] += 1
        if addr_exact:
            match_types['exact_addr'] += 1
        if addr_empty:
            match_types['addr_empty'] += 1
        if not name_exact and addr_exact:
            match_types['name_diff_addr_same'] += 1

print("Total evaluated pairs:", count)
for k, v in match_types.items():
    print(f"  {k}: {v} ({v/count:.2%})")
