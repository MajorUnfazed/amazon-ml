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
from eval.metric import compute_macro_f05

print("=" * 70)
print("EVALUATING MULTI-CHANNEL RETRIEVER + LIGHTGBM MODEL ON GT VALIDATION")
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

# 2. Load ALL true partners + 150,000 background partners
partner_lookup = {}
s1_lookup = {}
for eid, name, addr, c in s1_records:
    norm = normalize_record(name, addr, c)
    norm['entity_id'] = eid
    norm['country'] = c
    s1_lookup[eid] = norm

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

# 3. Build Multi-Channel High-Recall Index
word_df = defaultdict(int)
char3_df = defaultdict(int)

for pid, norm in partner_lookup.items():
    n_clean = str(norm["name_clean"])
    n_core = str(norm["name_core"])
    a_clean = str(norm["address_clean"])
    for w in set(n_clean.split()):
        if len(w) >= 3:
            word_df[w] += 1
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
    a_clean = str(norm["address_clean"])
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

# 4. Generate Top-25 Candidates
pair_rows = []
for eid, norm in s1_lookup.items():
    cand_scores = defaultdict(float)
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        for pid in slug_index.get(slug, []):
            cand_scores[pid] += 30.0

    nums = norm.get("building_numbers", [])
    a_toks = [t for t in str(norm["address_clean"]).split() if len(t) >= 3 and not t.isdigit()]
    if nums and a_toks:
        addr_k = f"{nums[0]}_{a_toks[0]}"
        pids = addr_key_index.get(addr_k, [])
        if 0 < len(pids) <= 300:
            for pid in pids:
                cand_scores[pid] += 25.0

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

    core_flat = str(norm["name_core"]).replace(" ", "")
    c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
    for c3 in c3_set:
        idf = char3_idf.get(c3, 0.0)
        if idf > 3.0:
            pids = char3_index.get(c3, [])
            if 0 < len(pids) <= 1000:
                for pid in pids:
                    cand_scores[pid] += idf * 0.4

    pc = norm.get("postal_code")
    if pc:
        for pid in postal_index.get(pc, []):
            if pid in cand_scores:
                cand_scores[pid] += 5.0

    if cand_scores:
        top_cands = heapq.nlargest(25, cand_scores.items(), key=lambda x: x[1])
        for pid, score in top_cands:
            pair_rows.append({
                "source1_entity_id": eid,
                "partner_entity_id": pid,
                "max_retriever_score": float(score),
                "score_name_tfidf": float(score) / 10.0,
                "score_name_addr_tfidf": 0.0,
                "score_addr_tfidf": 0.0,
                "score_postal_block": 0.0,
                "score_acronym_block": 0.0
            })

cand_df = pd.DataFrame(pair_rows)
print(f"Generated {len(cand_df):,} candidate pairs.")

# 5. Model Inference
artifacts_dir = "artifacts"
model = joblib.load(os.path.join(artifacts_dir, "lgb_matcher.joblib"))
calibrator = joblib.load(os.path.join(artifacts_dir, "calibrator.joblib"))
fb: FeatureBuilder = joblib.load(os.path.join(artifacts_dir, "feature_builder.joblib"))
feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))

feat_df = fb.build_features(cand_df, s1_lookup, partner_lookup=partner_lookup)
X = feat_df[feature_cols]
p_raw = model.predict_proba(X)[:, 1]
p_cal = calibrator.transform(p_raw)
feat_df["p_cal"] = p_cal

resolved = resolve_one_to_one(feat_df, prob_col="p_cal")

print("\n" + "=" * 70)
print(f"EVALUATING MACRO F0.5 SCORES ACROSS THRESHOLDS:")
for th in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
    p_pass = resolved[resolved["p_cal"] >= th]
    d = defaultdict(set)
    for s1, p in zip(p_pass["source1_entity_id"], p_pass["partner_entity_id"]):
        d[s1].add(p)
    p_eval = {eid: d.get(eid, set()) for eid, _, _, _ in s1_records}
    sc = compute_macro_f05(eval_s1_dict, p_eval)
    
    # Calculate TP, Precision, Recall
    all_pred = sum(len(v) for v in p_eval.values())
    all_tp = sum(len(p_eval[eid] & eval_s1_dict[eid]) for eid, _, _, _ in s1_records)
    prec = all_tp / all_pred if all_pred else 1.0
    rec = all_tp / total_true_matches
    print(f"  Threshold {th:.2f} -> Macro F0.5 = {sc:.4f} (Global Prec: {prec:.2%}, Rec: {rec:.2%}, Matches: {all_pred:,})")
print("=" * 70)
