"""Fast, aligned training and validation pipeline for LinkSure."""

import os
import sys
import time
import random
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

# Add src to sys.path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize.text import normalize_record
from features.builder import FeatureBuilder
from decide.one_to_one import resolve_one_to_one
from decide.expected_f import decide_global_threshold
from eval.metric import evaluate_predictions, compute_blocking_metrics

random.seed(42)
np.random.seed(42)

train_dir = os.path.join("dataset", "train")
artifacts_dir = os.path.join("artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

print("=" * 70, flush=True)
print("LinkSure: Model Training & Validation Protocol", flush=True)
print("=" * 70, flush=True)

# 1. Fast load of ground truth and corresponding S1 entities
N_TARGET_S1 = 100000
print(f"\n[1/6] Loading {N_TARGET_S1:,} training S1 entities and ground truth...", flush=True)
t0 = time.time()

needed_s1 = set()
sample_gt = {}

with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f_gt:
    next(f_gt)
    for line in f_gt:
        p = line.rstrip("\r\n").split("\t")
        s1 = p[0]
        matches = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) > 1 and p[1].strip() else set()
        sample_gt[s1] = matches
        needed_s1.add(s1)
        if len(needed_s1) >= N_TARGET_S1:
            break

target_partners = {pid for plist in sample_gt.values() for pid in plist}

# Find S1 records
sample_s1_dict = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f_s1:
    next(f_s1)
    for line in f_s1:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            sample_s1_dict[p[0]] = (p[1], p[2], p[3])
            if len(sample_s1_dict) == len(needed_s1):
                break

print(f"  Loaded {len(sample_s1_dict):,} S1 queries with {len(target_partners):,} true partner matches in {time.time() - t0:.2f}s.", flush=True)

# Normalize S1 queries
norm_s1_records = []
for s1_id, (n, a, c) in sample_s1_dict.items():
    norm = normalize_record(n, a, c)
    norm["entity_id"] = s1_id
    norm["country"] = c
    norm_s1_records.append(norm)

s1_df = pd.DataFrame(norm_s1_records)

# 2. Fast load partner records (all target partners + background partners)
print("\n[2/6] Loading partner candidate pool...", flush=True)
t0 = time.time()
found_partners = {}

for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            pid = p[0]
            if pid in target_partners or (len(found_partners) < 500000 and random.random() < 0.50):
                found_partners[pid] = (p[1], p[2], p[3])

partner_records = []
for pid, (n, a, c) in found_partners.items():
    norm = normalize_record(n, a, c)
    norm["entity_id"] = pid
    norm["country"] = c
    partner_records.append(norm)

partner_df = pd.DataFrame(partner_records)
print(f"  Loaded {len(partner_df):,} partner records in {time.time() - t0:.2f}s (Captured {len(target_partners & set(found_partners)):,} / {len(target_partners):,} target partners).", flush=True)

# 3. Candidate Generation (Blocking)
print("\n[3/6] Generating candidate pairs via partitioned multi-signal inverted index...", flush=True)
t0 = time.time()
pair_rows = []

for country in s1_df["country"].unique():
    c_s1 = s1_df[s1_df["country"] == country]
    c_p = partner_df[partner_df["country"] == country]

    # Inverted token index & slug index
    token_index = defaultdict(list)
    slug_index = defaultdict(list)

    for pid, n_clean, n_core, a_clean in zip(c_p["entity_id"], c_p["name_clean"], c_p["name_core"], c_p["address_clean"]):
        toks = set(str(n_clean).split() + str(a_clean).split()[:3])
        for t in toks:
            if len(t) >= 2:
                token_index[t].append(pid)
        slug = str(n_core).replace(" ", "")[:25]
        if len(slug) >= 4:
            slug_index[slug].append(pid)

    # Candidate retrieval per S1
    for s1_id, n_clean, n_core, a_clean in zip(c_s1["entity_id"], c_s1["name_clean"], c_s1["name_core"], c_s1["address_clean"]):
        cand_scores = defaultdict(float)

        slug = str(n_core).replace(" ", "")[:25]
        if len(slug) >= 4:
            for pid in slug_index.get(slug, []):
                cand_scores[pid] += 8.0

        toks = set(str(n_clean).split() + str(a_clean).split()[:3])
        for t in toks:
            pids = token_index.get(t, [])
            if 0 < len(pids) <= 500:
                w = 3.0 if t in str(n_clean) else 1.0
                for pid in pids:
                    cand_scores[pid] += w

        if cand_scores:
            top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:50]
            for pid, score in top_cands:
                pair_rows.append({
                    "source1_entity_id": s1_id,
                    "partner_entity_id": pid,
                    "max_retriever_score": float(score),
                    "score_name_tfidf": float(score) / 10.0,
                    "score_name_addr_tfidf": 0.0,
                    "score_addr_tfidf": 0.0,
                    "score_postal_block": 0.0,
                    "score_acronym_block": 0.0
                })

cands_df = pd.DataFrame(pair_rows)
print(f"  Generated {len(cands_df):,} candidate pairs in {time.time() - t0:.2f}s.", flush=True)

# Blocking audit
s1_list = s1_df["entity_id"].tolist()
cands_dict = {s1: set() for s1 in s1_list}
for s1_id, pid in zip(cands_df["source1_entity_id"], cands_df["partner_entity_id"]):
    cands_dict[s1_id].add(pid)

audit = compute_blocking_metrics(sample_gt, cands_dict, total_pool_size=len(partner_df))
print(f"  Blocking Audit: Recall Ceiling = {audit['recall_ceiling']:.2%}, Avg Candidates/S1 = {audit['avg_candidates_per_s1']:.1f}", flush=True)

# 4. Feature Extraction
print("\n[4/6] Extracting relational pair features...", flush=True)
t0 = time.time()
fb = FeatureBuilder()
all_texts = s1_df["name_clean"].tolist() + partner_df["name_clean"].tolist()
fb.fit_idf(all_texts)

features_df = fb.build_features(cands_df, s1_df, partner_df)

# Label pairs
true_pair_set = {(s1, pid) for s1, plist in sample_gt.items() for pid in plist}
features_df["label"] = [
    1 if (s1, pid) in true_pair_set else 0
    for s1, pid in zip(features_df["source1_entity_id"], features_df["partner_entity_id"])
]

feature_cols = [c for c in features_df.columns if c not in {"source1_entity_id", "partner_entity_id", "label"}]
print(f"  Extracted {len(feature_cols)} features for {len(features_df):,} pairs in {time.time() - t0:.2f}s.", flush=True)
print(f"  Positive pairs: {features_df['label'].sum():,} ({features_df['label'].mean():.2%})", flush=True)

# 5. Model Training, 5-Fold GroupKFold, & Calibration
print("\n[5/6] Training LightGBM with 5-Fold GroupKFold & Isotonic Calibration...", flush=True)
t0 = time.time()

groups = features_df["source1_entity_id"].values
X = features_df[feature_cols].copy()
y = features_df["label"].values.astype(int)

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(features_df), dtype=float)
importances = np.zeros(len(feature_cols), dtype=float)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    importances += clf.feature_importances_ / 5.0

print(f"  CV complete in {time.time() - t0:.2f}s.", flush=True)

# Fit Isotonic Calibrator
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(oof_preds, y)
oof_cal = calibrator.transform(oof_preds)

features_df["p_raw"] = oof_preds
features_df["p_cal"] = oof_cal

# Top features
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
print("\n  Top 10 Most Important Features:", flush=True)
for idx, (fn, imp) in enumerate(feat_imp[:10], 1):
    print(f"    {idx:2d}. {fn:<30}: {imp:6.1f}", flush=True)

# 6. Evaluate Decision Strategies & Save Model
print("\n[6/6] Evaluating Decision Strategies on Out-Of-Fold Validation Data...", flush=True)

# Enforce 1-to-1 consistency
features_1to1 = resolve_one_to_one(features_df, prob_col="p_cal")

# Strategy evaluation across thresholds
best_f05 = 0.0
best_thresh = 0.45

for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70]:
    preds = decide_global_threshold(features_1to1, s1_list, prob_col="p_cal", threshold=thresh)
    res = evaluate_predictions(sample_gt, preds)
    print(f"  1-to-1 + Thresh {thresh:.2f}: Macro F0.5 = {res['macro_f05']:.4f} (Prec = {res['mean_precision']:.4f}, Rec = {res['mean_recall']:.4f}, Sing Acc = {res['singleton_accuracy']:.2%})", flush=True)
    if res['macro_f05'] > best_f05:
        best_f05 = res['macro_f05']
        best_thresh = thresh

print(f"\nOptimal Decision Rule: 1-to-1 Consistency + Threshold {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})", flush=True)

# Fit final full model and save artifacts
print("\nSaving final trained model, calibrator, and feature builder to artifacts/...", flush=True)
final_model = lgb.LGBMClassifier(**params)
final_model.fit(X, y)

joblib.dump(final_model, os.path.join(artifacts_dir, "lgb_matcher.joblib"))
joblib.dump(calibrator, os.path.join(artifacts_dir, "calibrator.joblib"))
joblib.dump(fb, os.path.join(artifacts_dir, "feature_builder.joblib"))
joblib.dump(feature_cols, os.path.join(artifacts_dir, "feature_cols.joblib"))
joblib.dump(best_thresh, os.path.join(artifacts_dir, "best_threshold.joblib"))

print("Artifacts successfully saved:", flush=True)
print(f"  - {os.path.join(artifacts_dir, 'lgb_matcher.joblib')}", flush=True)
print(f"  - {os.path.join(artifacts_dir, 'calibrator.joblib')}", flush=True)
print(f"  - {os.path.join(artifacts_dir, 'feature_builder.joblib')}", flush=True)
print(f"  - {os.path.join(artifacts_dir, 'feature_cols.joblib')}", flush=True)
print(f"  - {os.path.join(artifacts_dir, 'best_threshold.joblib')}", flush=True)
print("\n" + "=" * 70, flush=True)
print("TRAINING & VALIDATION COMPLETE", flush=True)
print("=" * 70, flush=True)
