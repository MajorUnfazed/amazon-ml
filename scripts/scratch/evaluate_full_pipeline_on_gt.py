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
print("EVALUATING END-TO-END PIPELINE ON GROUND TRUTH VALIDATION SET")
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

# 3. Build index
token_index = defaultdict(list)
slug_index = defaultdict(list)
for pid, norm in partner_lookup.items():
    toks = set(str(norm["name_clean"]).split() + str(norm["address_clean"]).split()[:3])
    for t in toks:
        if len(t) >= 2:
            token_index[t].append(pid)
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        slug_index[slug].append(pid)

# 4. Generate candidates
pair_rows = []
for eid, norm in s1_lookup.items():
    cand_scores = defaultdict(float)
    slug = str(norm["name_core"]).replace(" ", "")[:25]
    if len(slug) >= 4:
        for pid in slug_index.get(slug, []):
            cand_scores[pid] += 8.0
    toks = set(str(norm["name_clean"]).split() + str(norm["address_clean"]).split()[:3])
    for t in toks:
        pids = token_index.get(t, [])
        if 0 < len(pids) <= 300:
            w = 3.0 if t in str(norm["name_clean"]) else 1.0
            for pid in pids:
                cand_scores[pid] += w
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
print(f"Generated {len(cand_df):,} candidate pairs for {len(s1_records)} queries.")

# 5. Extract features & score
artifacts_dir = "artifacts"
model = joblib.load(os.path.join(artifacts_dir, "lgb_matcher.joblib"))
calibrator = joblib.load(os.path.join(artifacts_dir, "calibrator.joblib"))
fb: FeatureBuilder = joblib.load(os.path.join(artifacts_dir, "feature_builder.joblib"))
feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
threshold = joblib.load(os.path.join(artifacts_dir, "best_threshold.joblib"))

feat_df = fb.build_features(cand_df, s1_lookup, partner_lookup=partner_lookup)
X = feat_df[feature_cols]
p_raw = model.predict_proba(X)[:, 1]
p_cal = calibrator.transform(p_raw)
feat_df["p_cal"] = p_cal

# Resolve 1-to-1
resolved = resolve_one_to_one(feat_df, prob_col="p_cal")

# Predict with threshold
passing = resolved[resolved["p_cal"] >= threshold]
preds_dict = defaultdict(set)
for s1_id, p_id in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
    preds_dict[s1_id].add(p_id)

preds_for_eval = {eid: preds_dict.get(eid, set()) for eid, _, _, _ in s1_records}
score = compute_macro_f05(eval_s1_dict, preds_for_eval)

print("\n" + "=" * 70)
print(f"VALIDATION PERFORMANCE ON GT SAMPLE:")
print(f"  Macro F0.5 Score: {score:.4f}")
print(f"  Threshold used:   {threshold:.2f}")
print("=" * 70)

# Check scores at different thresholds
for th in [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70]:
    p_pass = resolved[resolved["p_cal"] >= th]
    d = defaultdict(set)
    for s1, p in zip(p_pass["source1_entity_id"], p_pass["partner_entity_id"]):
        d[s1].add(p)
    p_eval = {eid: d.get(eid, set()) for eid, _, _, _ in s1_records}
    sc = compute_macro_f05(eval_s1_dict, p_eval)
    print(f"  Threshold {th:.2f} -> Macro F0.5 = {sc:.4f}")
