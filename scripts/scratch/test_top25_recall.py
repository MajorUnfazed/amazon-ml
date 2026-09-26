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

print("=" * 70)
print("TESTING TOP-25 CANDIDATE RECALL WITH MULTI-CHANNEL SCORING")
print("=" * 70)

# 1. Load 1,000 US S1 queries from train
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
total_true = sum(len(v) for v in eval_s1_dict.values())

# 2. Load partners
partner_lookup = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4 and p[3] == 'US':
                pid = p[0]
                if pid in all_true_partners or (len(partner_lookup) < 150000 and random.random() < 0.05):
                    norm = normalize_record(p[1], p[2], p[3])
                    norm['entity_id'] = pid
                    norm['country'] = 'US'
                    partner_lookup[pid] = norm

# 3. Build index with word and char3 IDF
word_df = defaultdict(int)
char3_df = defaultdict(int)

for pid, norm in partner_lookup.items():
    n_clean = str(norm["name_clean"])
    n_core = str(norm["name_core"])
    a_clean = str(norm["address_clean"])
    for w in set(n_clean.split()):
        if len(w) >= 3:
            word_df[w] += 1
    # Distinct address words (street/city)
    for w in set(a_clean.split()):
        if len(w) >= 4 and not w.isdigit():
            word_df[w] += 1
    core_flat = n_core.replace(" ", "")
    for i in range(len(core_flat) - 2):
        char3_df[core_flat[i:i+3]] += 1

N_docs = len(partner_lookup)
word_idf = {w: math.log(1.0 + N_docs / df) for w, df in word_df.items() if df <= 25000}
char3_idf = {c3: math.log(1.0 + N_docs / df) for c3, df in char3_df.items() if df <= 25000}

slug_index = defaultdict(list)
addr_key_index = defaultdict(list)
word_index = defaultdict(list)
char3_index = defaultdict(list)
postal_index = defaultdict(list)

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
        if w in word_idf:
            word_index[w].append(pid)
    for w in set(a_clean.split()):
        if w in word_idf and len(w) >= 4 and not w.isdigit():
            word_index[w].append(pid)
    core_flat = n_core.replace(" ", "")
    c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
    for c3 in c3_set:
        if c3 in char3_idf:
            char3_index[c3].append(pid)
    pc = norm.get("postal_code")
    if pc:
        postal_index[pc].append(pid)

print(f"Index built for {len(partner_lookup):,} partner records.")

# 4. Score queries
for K in [15, 20, 25, 30]:
    hits = 0
    t0 = time.time()
    for eid, name, addr, c in s1_records:
        norm = normalize_record(name, addr, c)
        true_set = eval_s1_dict[eid]
        cand_scores = defaultdict(float)

        # 1. Exact core slug (highest weight)
        slug = str(norm["name_core"]).replace(" ", "")[:25]
        if len(slug) >= 4:
            for pid in slug_index.get(slug, []):
                cand_scores[pid] += 30.0

        # 2. Address key (number + street)
        nums = norm.get("building_numbers", [])
        a_toks = [t for t in str(norm["address_clean"]).split() if len(t) >= 3 and not t.isdigit()]
        if nums and a_toks:
            addr_k = f"{nums[0]}_{a_toks[0]}"
            pids = addr_key_index.get(addr_k, [])
            if 0 < len(pids) <= 300:
                for pid in pids:
                    cand_scores[pid] += 25.0

        # 3. Clean Words (IDF) - Name and Address
        for w in set(str(norm["name_clean"]).split()):
            idf = word_idf.get(w, 0.0)
            if idf > 2.0:
                pids = word_index.get(w, [])
                if 0 < len(pids) <= 1500:
                    for pid in pids:
                        cand_scores[pid] += idf

        for w in set(str(norm["address_clean"]).split()):
            if len(w) >= 4 and not w.isdigit():
                idf = word_idf.get(w, 0.0)
                if idf > 2.5:
                    pids = word_index.get(w, [])
                    if 0 < len(pids) <= 1500:
                        for pid in pids:
                            cand_scores[pid] += idf * 0.7

        # 4. Character 3-grams
        core_flat = str(norm["name_core"]).replace(" ", "")
        c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
        for c3 in c3_set:
            idf = char3_idf.get(c3, 0.0)
            if idf > 3.0:
                pids = char3_index.get(c3, [])
                if 0 < len(pids) <= 1000:
                    for pid in pids:
                        cand_scores[pid] += idf * 0.4

        # 5. Postal code match (bonus if already matched name or address)
        pc = norm.get("postal_code")
        if pc:
            for pid in postal_index.get(pc, []):
                if pid in cand_scores:
                    cand_scores[pid] += 5.0

        if cand_scores:
            top_cands = set(x[0] for x in heapq.nlargest(K, cand_scores.items(), key=lambda x: x[1]))
        else:
            top_cands = set()

        hits += len(top_cands.intersection(true_set))

    elapsed = time.time() - t0
    recall = hits / total_true
    print(f"Top-{K:02d} Recall: {recall:.2%} ({hits:,} / {total_true:,}) | Speed: {len(s1_records)/elapsed:.1f} q/s")
