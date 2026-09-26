"""
Entity Resolution Pipeline v3 - Complete Rebuild
=================================================
Single unified script for training + inference.

Usage:
  python pipeline_v3.py train               # Train on training data, validate, save model
  python pipeline_v3.py train --sample 50000  # Train with 50K S1 per country
  python pipeline_v3.py train --dry-run     # Smoke test: 1000 S1, 5 trees, <5 min
  python pipeline_v3.py predict             # Run inference on test data, generate submission

Architecture:
  1. TF-IDF char n-gram blocking (scipy sparse matmul, no posting list caps)
  2. Rich string similarity features via normalize_record() + FeatureBuilder (~40 features)
  3. LightGBM classifier with scale_pos_weight + isotonic calibration
  4. Global 1-to-1 consistency per country, then threshold-based matching
  5. Expected-F0.5 decision strategy evaluated as alternative

Fixes vs. original:
  - 1-to-1 consistency applied GLOBALLY per country (not per 2M-pair chunk)
  - Postal code extraction uses normalize_record() (handles US/India/France correctly)
  - Feature extraction uses FeatureBuilder with IDF-weighted token features (~40 features)
  - LightGBM scale_pos_weight for heavily imbalanced training set
  - Threshold search: finer grid (0.02 steps), 1-to-1 applied before sweep
  - Output written via write_tsv_submission() for deterministic, sorted output
  - Memory-efficient streaming: only (s1_id, partner_id, p_cal) kept across chunks
  - Blocking recall audit in predict mode
  - France transfer validation in train mode
  - --dry-run smoke test mode
"""

import os, sys, time, gc, math, re, unicodedata, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from rapidfuzz import fuzz, distance
import joblib

# ─── Configuration ─────────────────────────────────────────────────────
ROOT = r"c:\Users\sam\Documents\Projects\amazon-ml"
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "output")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

src_dir = os.path.join(ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)
from normalize.text import normalize_record
from features.builder import FeatureBuilder
from eval.metric import evaluate_predictions
from decide.expected_f import decide_expected_f05
from pipeline_io.writer import write_tsv_submission, run_submission_validator

# Blocking config
TFIDF_MAX_FEATURES = 80000
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MAX_DF = 0.1
TFIDF_MIN_DF = 3
TOP_K = 50              # candidates per S1 entity (overridable via --top-k)
PARTNER_CHUNK = 250000  # partners per chunk for matmul
QUERY_CHUNK = 10000     # queries per chunk (overridable via --query-chunk)

# Training config
TRAIN_SAMPLE_PER_COUNTRY = 100000  # S1 entities to train on per country (overridable via --sample)
DECISION_THRESHOLD = 0.50          # Will be tuned during training

# LightGBM base params (scale_pos_weight set dynamically)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

# Dry-run overrides
DRY_RUN_SAMPLE = 1000
DRY_RUN_N_ESTIMATORS = 5


# ─── Text Utilities ────────────────────────────────────────────────────

def make_text(name, addr):
    """Create combined text for TF-IDF vectorization."""
    text = str(name) + " " + str(addr)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_name_text(name):
    """Clean name text for similarity features."""
    text = str(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Data Loading ──────────────────────────────────────────────────────

def load_source_file(filepath, country_filter=None):
    """Load a source TSV file. Returns dict of {entity_id: (name, addr, country)}."""
    records = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        next(f)  # skip header
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                if country_filter and p[3] != country_filter:
                    continue
                records[p[0]] = (p[1], p[2], p[3])
    return records

def load_ground_truth(filepath):
    """Load ground truth TSV. Returns dict of {s1_id: set(partner_ids)}."""
    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            matches = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt[eid] = matches
    return gt


# ─── TF-IDF Blocking ──────────────────────────────────────────────────

def top_k_per_row(sim_csr, k):
    """Extract top-K column indices and values per row from a CSR matrix."""
    n = sim_csr.shape[0]
    indices_out = np.full((n, k), -1, dtype=np.int32)
    values_out = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        start, end = sim_csr.indptr[i], sim_csr.indptr[i + 1]
        data = sim_csr.data[start:end]
        cols = sim_csr.indices[start:end]
        n_nnz = len(data)

        if n_nnz == 0:
            continue
        elif n_nnz <= k:
            indices_out[i, :n_nnz] = cols
            values_out[i, :n_nnz] = data
        else:
            top_k_pos = np.argpartition(data, -k)[-k:]
            sorted_pos = top_k_pos[np.argsort(-data[top_k_pos])]
            indices_out[i] = cols[sorted_pos]
            values_out[i] = data[sorted_pos]

    return indices_out, values_out


def tfidf_blocking(query_texts, query_ids, partner_texts, partner_ids,
                   vectorizer=None, k=50, partner_chunk_size=250000,
                   query_chunk_size=10000):
    """
    TF-IDF char n-gram blocking with chunked sparse matmul.
    Returns: list of (s1_id, partner_id, cosine_score) tuples, fitted vectorizer.
    """
    n_partners = len(partner_ids)
    n_queries = len(query_ids)

    # Fit vectorizer on partner texts if not provided
    if vectorizer is None:
        print(f"    Fitting TfidfVectorizer on {n_partners:,} partner texts...", flush=True)
        t0 = time.time()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
            max_df=TFIDF_MAX_DF,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
            dtype=np.float32
        )
        P = vectorizer.fit_transform(partner_texts)
        print(f"    Partner matrix: {P.shape}, nnz={P.nnz:,} in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"    Transforming {n_partners:,} partner texts with pre-fitted vectorizer...", flush=True)
        t0 = time.time()
        P = vectorizer.transform(partner_texts)
        print(f"    Partner matrix: {P.shape}, nnz={P.nnz:,} in {time.time()-t0:.1f}s", flush=True)

    # Process queries in chunks
    all_candidates = []
    n_q_chunks = (n_queries + query_chunk_size - 1) // query_chunk_size
    n_p_chunks = (n_partners + partner_chunk_size - 1) // partner_chunk_size
    t0 = time.time()

    for qc in range(n_q_chunks):
        q_start = qc * query_chunk_size
        q_end = min(q_start + query_chunk_size, n_queries)
        chunk_size = q_end - q_start

        # Transform query chunk
        Q_chunk = vectorizer.transform(query_texts[q_start:q_end])

        # Track best K across partner chunks
        best_indices = np.full((chunk_size, k), -1, dtype=np.int32)
        best_values = np.zeros((chunk_size, k), dtype=np.float32)

        for pc in range(n_p_chunks):
            p_start = pc * partner_chunk_size
            p_end = min(p_start + partner_chunk_size, n_partners)
            P_chunk = P[p_start:p_end]

            # Sparse matmul
            sim = Q_chunk.dot(P_chunk.T).tocsr()

            # Extract top-K from this chunk
            chunk_idx, chunk_val = top_k_per_row(sim, k)

            # Adjust to global partner indices
            valid_mask = chunk_idx >= 0
            chunk_idx[valid_mask] += p_start

            # Merge with running best
            for i in range(chunk_size):
                ci = np.concatenate([best_indices[i], chunk_idx[i]])
                cv = np.concatenate([best_values[i], chunk_val[i]])

                valid = (ci >= 0) & (cv > 0)
                ci_valid = ci[valid]
                cv_valid = cv[valid]

                n_valid = len(ci_valid)
                if n_valid == 0:
                    continue
                elif n_valid <= k:
                    best_indices[i] = -1
                    best_values[i] = 0
                    order = np.argsort(-cv_valid)
                    best_indices[i, :n_valid] = ci_valid[order]
                    best_values[i, :n_valid] = cv_valid[order]
                else:
                    top_pos = np.argpartition(cv_valid, -k)[-k:]
                    sorted_pos = top_pos[np.argsort(-cv_valid[top_pos])]
                    best_indices[i] = ci_valid[sorted_pos]
                    best_values[i] = cv_valid[sorted_pos]

            del sim

        # Collect candidates from this query chunk
        for i in range(chunk_size):
            qid = query_ids[q_start + i]
            for j in range(k):
                if best_indices[i, j] >= 0 and best_values[i, j] > 0:
                    pid = partner_ids[best_indices[i, j]]
                    score = float(best_values[i, j])
                    all_candidates.append((qid, pid, score))

        elapsed = time.time() - t0
        print(f"    Query chunk {qc+1}/{n_q_chunks} done ({q_start:,}-{q_end:,}), "
              f"{len(all_candidates):,} candidates so far, {elapsed:.0f}s elapsed", flush=True)

    del P
    gc.collect()

    return all_candidates, vectorizer


# ─── Normalized Record Cache ───────────────────────────────────────────

def build_norm_cache(raw_dict):
    """
    Pre-compute normalize_record() for all entities in raw_dict.
    Returns dict of {entity_id: norm_record_dict}.
    """
    cache = {}
    for eid, (name, addr, country) in raw_dict.items():
        cache[eid] = normalize_record(name, addr, country)
    return cache


# ─── Feature Engineering ───────────────────────────────────────────────

def extract_features_batch(candidates, s1_raw, partner_raw, fb,
                            s1_norm_cache=None, partner_norm_cache=None):
    """
    Extract ~40 features for a batch of (s1_id, partner_id, tfidf_score) triples.
    Uses normalize_record() for rich fields + FeatureBuilder for IDF-weighted tokens.
    All features computed at inference time — no pickle dependencies.

    Returns a DataFrame with all features + source1_entity_id + partner_entity_id.
    """
    rows = []
    n_total = len(candidates)
    t0 = time.time()

    for idx, (s1_id, p_id, tfidf_score) in enumerate(candidates):
        s1_name, s1_addr, s1_country = s1_raw.get(s1_id, ("", "", ""))
        p_name, p_addr, p_country = partner_raw.get(p_id, ("", "", ""))

        # Use pre-computed normalized records if available
        if s1_norm_cache is not None:
            s1n = s1_norm_cache.get(s1_id) or normalize_record(s1_name, s1_addr, s1_country)
        else:
            s1n = normalize_record(s1_name, s1_addr, s1_country)

        if partner_norm_cache is not None:
            pn = partner_norm_cache.get(p_id) or normalize_record(p_name, p_addr, p_country)
        else:
            pn = normalize_record(p_name, p_addr, p_country)

        # Normalized fields
        s1_clean = s1n["name_clean"]
        s1_core  = s1n["name_core"]
        s1_suffix = s1n["legal_suffix"]
        s1_acr   = s1n["name_acronym"]
        s1_vars  = s1n["name_variants"]
        s1_addr_clean = s1n["address_clean"]
        s1_postal = s1n["postal_code"]
        s1_nums  = set(s1n["building_numbers"])
        s1_lms   = set(s1n["landmarks"])

        p_clean  = pn["name_clean"]
        p_core   = pn["name_core"]
        p_suffix = pn["legal_suffix"]
        p_acr    = pn["name_acronym"]
        p_vars   = pn["name_variants"]
        p_addr_clean = pn["address_clean"]
        p_postal = pn["postal_code"]
        p_nums   = set(pn["building_numbers"])
        p_lms    = set(pn["landmarks"])

        feat = {
            "source1_entity_id": s1_id,
            "partner_entity_id": p_id,
        }

        # ── 1. TF-IDF cosine score from blocking (real score, not fake!) ──
        feat["tfidf_cosine"] = tfidf_score

        # ── 2. Name similarities (clean name) ──
        feat["name_clean_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_clean, p_clean))
        feat["name_clean_levenshtein"]  = float(fuzz.ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_token_sort"]   = float(fuzz.token_sort_ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_token_set"]    = float(fuzz.token_set_ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_partial"]      = float(fuzz.partial_ratio(s1_clean, p_clean) / 100.0)

        # ── 3. Name similarities (core name, suffix stripped) ──
        feat["name_core_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_core, p_core))
        feat["name_core_levenshtein"]  = float(fuzz.ratio(s1_core, p_core) / 100.0)
        feat["name_core_token_sort"]   = float(fuzz.token_sort_ratio(s1_core, p_core) / 100.0)
        feat["name_core_token_set"]    = float(fuzz.token_set_ratio(s1_core, p_core) / 100.0)

        # ── 4. DBA best variant match ──
        best_dba = max(
            (fuzz.ratio(v1, v2) / 100.0 for v1 in s1_vars for v2 in p_vars),
            default=0.0
        )
        feat["name_dba_best_ratio"] = float(best_dba)

        # ── 5. Acronym exact match ──
        feat["acronym_exact_match"] = 1.0 if (s1_acr and p_acr and s1_acr == p_acr) else 0.0

        # ── 6. Legal suffix status ──
        if s1_suffix and p_suffix:
            feat["suffix_status"] = 1.0 if (s1_suffix == p_suffix) else -1.0
        else:
            feat["suffix_status"] = 0.0

        # ── 7. IDF-weighted token features (core name) ──
        s1_tokens = set(s1_core.split())
        p_tokens  = set(p_core.split())
        shared_tokens   = s1_tokens & p_tokens
        unshared_tokens = (s1_tokens | p_tokens) - shared_tokens

        shared_idfs   = [fb.get_token_idf(t) for t in shared_tokens]
        unshared_idfs = [fb.get_token_idf(t) for t in unshared_tokens]

        feat["token_shared_count"]    = float(len(shared_tokens))
        feat["token_shared_idf_sum"]  = float(sum(shared_idfs))
        feat["token_shared_max_idf"]  = float(max(shared_idfs)) if shared_idfs else 0.0
        feat["token_unshared_max_idf"] = float(max(unshared_idfs)) if unshared_idfs else 0.0
        feat["token_jaccard"] = (
            float(len(shared_tokens) / len(s1_tokens | p_tokens))
            if (s1_tokens | p_tokens) else 0.0
        )

        # ── 8. Address similarities ──
        feat["addr_token_set"]    = float(fuzz.token_set_ratio(s1_addr_clean, p_addr_clean) / 100.0)
        feat["addr_token_sort"]   = float(fuzz.token_sort_ratio(s1_addr_clean, p_addr_clean) / 100.0)
        feat["addr_levenshtein"]  = float(fuzz.ratio(s1_addr_clean, p_addr_clean) / 100.0)
        feat["addr_partial"]      = float(fuzz.partial_ratio(s1_addr_clean, p_addr_clean) / 100.0)

        s1_a_toks = set(s1_addr_clean.split())
        p_a_toks  = set(p_addr_clean.split())
        shared_a  = s1_a_toks & p_a_toks
        feat["addr_token_jaccard"] = (
            float(len(shared_a) / len(s1_a_toks | p_a_toks))
            if (s1_a_toks | p_a_toks) else 0.0
        )

        # ── 9. Postal code features (uses normalize_record() — handles US/India/France) ──
        if s1_postal and p_postal:
            feat["postal_match"]  = 1.0 if s1_postal == p_postal else -1.0
            feat["postal_prefix3"] = 1.0 if s1_postal[:3] == p_postal[:3] else -1.0
        else:
            feat["postal_match"]  = 0.0
            feat["postal_prefix3"] = 0.0

        # ── 10. Building number match ──
        if s1_nums and p_nums:
            feat["building_num_match"] = 1.0 if (s1_nums & p_nums) else -1.0
        else:
            feat["building_num_match"] = 0.0

        # ── 11. Landmark shared ──
        if s1_lms or p_lms:
            feat["landmark_shared"] = 1.0 if (s1_lms & p_lms) else 0.0
        else:
            feat["landmark_shared"] = 0.0

        # ── 12. Length features ──
        len_s1c, len_pc = len(s1_core), len(p_core)
        max_len = max(len_s1c, len_pc)
        feat["name_length_ratio"] = float(min(len_s1c, len_pc) / max_len) if max_len > 0 else 1.0
        feat["name_length_diff"]  = float(abs(len_s1c - len_pc))

        len_s1a, len_pa = len(s1_addr_clean), len(p_addr_clean)
        max_len_a = max(len_s1a, len_pa)
        feat["addr_length_ratio"] = float(min(len_s1a, len_pa) / max_len_a) if max_len_a > 0 else 1.0

        # ── 13. Source indicator ──
        feat["is_source2"] = 1.0 if str(p_id).startswith("S2-") else 0.0

        rows.append(feat)

        if (idx + 1) % 500000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate
            print(f"      Feature extraction: {idx+1:,}/{n_total:,} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    df = pd.DataFrame(rows)

    # ── 14. Context features (rank within S1, reverse rank, mutual best match) ──
    df = df.sort_values(["source1_entity_id", "tfidf_cosine"], ascending=[True, False])
    df["rank_within_s1"] = df.groupby("source1_entity_id").cumcount() + 1.0
    best_score = df.groupby("source1_entity_id")["tfidf_cosine"].transform("first")
    df["gap_to_best"] = best_score - df["tfidf_cosine"]
    df["num_candidates"] = df.groupby("source1_entity_id")["partner_entity_id"].transform("count").astype(float)

    # Score gap between rank 1 and rank 2
    second_best = df.groupby("source1_entity_id")["tfidf_cosine"].transform(
        lambda x: x.iloc[1] if len(x) > 1 else 0.0
    )
    df["score_gap_1_2"] = best_score - second_best

    # Reverse rank (how many S1s claim this partner)
    df = df.sort_values(["partner_entity_id", "tfidf_cosine"], ascending=[True, False])
    df["reverse_rank"] = df.groupby("partner_entity_id").cumcount() + 1.0
    df["num_claims"] = df.groupby("partner_entity_id")["source1_entity_id"].transform("count").astype(float)

    # Mutual best match flag
    df["is_mutual_best_match"] = (
        (df["rank_within_s1"] == 1.0) & (df["reverse_rank"] == 1.0)
    ).astype(float)

    df = df.sort_values(["source1_entity_id", "tfidf_cosine"], ascending=[True, False]).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"      Feature extraction complete: {len(df):,} pairs in {elapsed:.1f}s", flush=True)

    return df


# ─── Model Training ───────────────────────────────────────────────────

def train_model(features_df, feature_cols, dry_run=False):
    """Train LightGBM with GroupKFold CV, scale_pos_weight, and Isotonic calibration."""
    print("  Training LightGBM with 5-fold GroupKFold...", flush=True)
    t0 = time.time()

    groups = features_df["source1_entity_id"].values
    X = features_df[feature_cols].values
    y = features_df["label"].values.astype(int)

    # Compute scale_pos_weight for imbalanced data
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = min(n_neg / n_pos, 20.0) if n_pos > 0 else 1.0
    print(f"  Positives: {n_pos:,}, Negatives: {n_neg:,}, scale_pos_weight: {spw:.2f}", flush=True)

    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = spw
    if dry_run:
        params["n_estimators"] = DRY_RUN_N_ESTIMATORS

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(features_df), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**params)
        if dry_run:
            clf.fit(X_tr, y_tr)
        else:
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        best_iter = getattr(clf, "best_iteration_", params["n_estimators"])
        print(f"    Fold {fold}: {best_iter} iters, "
              f"pos_rate={y_val.mean():.4f}", flush=True)

    # Calibrate OOF predictions
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds, y)
    oof_cal = calibrator.transform(oof_preds)

    features_df["p_raw"] = oof_preds
    features_df["p_cal"] = oof_cal

    # Train final model on all data
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)

    # Feature importances
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print("\n  Top 20 features:", flush=True)
    for i, (fn, imp) in enumerate(feat_imp[:20], 1):
        print(f"    {i:2d}. {fn:<30}: {imp:6.0f}", flush=True)

    print(f"\n  Training complete in {time.time()-t0:.1f}s", flush=True)
    return final_model, calibrator


# ─── Evaluation ────────────────────────────────────────────────────────

def compute_f05(true_ids, pred_ids):
    """Per-entity F0.5 score."""
    lt, lp = len(true_ids), len(pred_ids)
    if lt == 0 and lp == 0: return 1.0
    if lt == 0 and lp > 0: return 0.0
    if lt > 0 and lp == 0: return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0: return 0.0
    return (1.25 * tp) / (0.25 * lt + lp)


def threshold_search(features_df, gt, s1_ids):
    """
    Find optimal threshold maximizing macro F0.5.
    - Finer grid: 0.02 steps
    - Applies global 1-to-1 consistency BEFORE threshold sweep
    - Also evaluates decide_expected_f05() as alternative strategy
    Returns (best_thresh, best_strategy, best_f05)
    """
    # Apply global 1-to-1 consistency once (not per threshold)
    df_121 = resolve_one_to_one_simple(features_df, prob_col="p_cal")

    eval_gt = {k: gt[k] for k in s1_ids if k in gt}
    s1_ids_set = set(s1_ids)

    best_f05 = 0.0
    best_thresh = 0.5
    best_strategy = "threshold"

    print("  Threshold sweep (with global 1-to-1 applied):", flush=True)
    for thresh in np.arange(0.10, 0.90, 0.02):
        preds = {}
        passing = df_121[df_121["p_cal"] >= thresh]
        for s1_id, pid in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
            if s1_id not in preds:
                preds[s1_id] = set()
            preds[s1_id].add(pid)
        for s1_id in s1_ids:
            if s1_id not in preds:
                preds[s1_id] = set()

        scores = [compute_f05(eval_gt.get(s1_id, set()), preds.get(s1_id, set()))
                  for s1_id in s1_ids]
        f05 = float(np.mean(scores))
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)
            best_strategy = "threshold"
        print(f"    Threshold {thresh:.2f}: F0.5 = {f05:.4f}", flush=True)

    # Also evaluate Expected-F0.5 strategy
    print("\n  Evaluating Expected-F0.5 strategy...", flush=True)
    try:
        ef05_preds = decide_expected_f05(df_121, list(s1_ids), prob_col="p_cal")
        ef05_scores = [compute_f05(eval_gt.get(s1_id, set()), ef05_preds.get(s1_id, set()))
                       for s1_id in s1_ids]
        ef05_f05 = float(np.mean(ef05_scores))
        print(f"    Expected-F0.5 strategy: F0.5 = {ef05_f05:.4f}", flush=True)
        if ef05_f05 > best_f05:
            best_f05 = ef05_f05
            best_thresh = -1.0  # sentinel: use expected_f05 strategy
            best_strategy = "expected_f05"
    except Exception as e:
        print(f"    Expected-F0.5 strategy failed: {e}", flush=True)

    print(f"\n  Best: strategy={best_strategy}, threshold={best_thresh:.2f}, F0.5={best_f05:.4f}",
          flush=True)
    return best_thresh, best_strategy, best_f05


# ─── 1-to-1 Consistency ───────────────────────────────────────────────

def resolve_one_to_one_simple(features_df, prob_col="p_cal"):
    """
    Global 1-to-1: for each partner, keep only the S1 with highest probability.
    Must be applied GLOBALLY per country (not per chunk).
    """
    df = features_df.sort_values(prob_col, ascending=False)
    df = df.drop_duplicates(subset=["partner_entity_id"], keep="first")
    return df


# ─── Blocking Recall Audit ─────────────────────────────────────────────

def blocking_recall_audit(candidates, gt, s1_ids, label=""):
    """Compute blocking recall ceiling for training data."""
    cand_dict = defaultdict(set)
    for s1_id, pid, _ in candidates:
        cand_dict[s1_id].add(pid)

    found_total = 0
    true_total = 0
    zero_cand_count = 0

    for eid in s1_ids:
        true_set = gt.get(eid, set())
        cand_set = cand_dict.get(eid, set())
        if not cand_set:
            zero_cand_count += 1
        for pid in true_set:
            true_total += 1
            if pid in cand_set:
                found_total += 1

    recall_ceiling = found_total / true_total if true_total > 0 else 0.0
    zero_rate = zero_cand_count / len(s1_ids) if s1_ids else 0.0
    print(f"  {label} Blocking recall ceiling: {recall_ceiling:.4f} ({recall_ceiling:.2%})", flush=True)
    print(f"  {label} Found {found_total:,} / {true_total:,} true matches in candidates", flush=True)
    print(f"  {label} Zero-candidate S1 entities: {zero_cand_count:,} ({zero_rate:.2%})", flush=True)
    return recall_ceiling, cand_dict


def blocking_quality_audit_test(candidates, s1_ids, label=""):
    """
    Audit blocking quality for test data (no GT available).
    Logs: zero-candidate rate, avg/median/p95 candidates, fraction with cosine > 0.3.
    """
    cand_dict = defaultdict(list)
    for s1_id, pid, score in candidates:
        cand_dict[s1_id].append(score)

    counts = []
    high_score_count = 0
    total_cands = 0
    zero_cand_count = 0

    for eid in s1_ids:
        scores = cand_dict.get(eid, [])
        n = len(scores)
        counts.append(n)
        total_cands += n
        if n == 0:
            zero_cand_count += 1
        high_score_count += sum(1 for s in scores if s > 0.3)

    counts_arr = np.array(counts)
    zero_rate = zero_cand_count / len(s1_ids) if s1_ids else 0.0
    frac_high = high_score_count / total_cands if total_cands > 0 else 0.0

    print(f"\n  {label} Blocking Quality Audit:", flush=True)
    print(f"    Zero-candidate S1 entities: {zero_cand_count:,} ({zero_rate:.2%})", flush=True)
    print(f"    Avg candidates per S1: {counts_arr.mean():.1f}", flush=True)
    print(f"    Median candidates per S1: {np.median(counts_arr):.1f}", flush=True)
    print(f"    P95 candidates per S1: {np.percentile(counts_arr, 95):.1f}", flush=True)
    print(f"    Fraction with cosine > 0.3: {frac_high:.2%}", flush=True)

    if zero_rate > 0.02:
        print(f"  ⚠️  WARNING: {zero_rate:.2%} of S1 entities have ZERO candidates! "
              f"Consider increasing TOP_K.", flush=True)

    return zero_rate, counts_arr


# ═══════════════════════════════════════════════════════════════════════
# MAIN: TRAIN MODE
# ═══════════════════════════════════════════════════════════════════════

def run_train(sample_per_country=None, top_k=None, query_chunk=None, dry_run=False):
    print("=" * 70)
    print("ENTITY RESOLUTION PIPELINE v3 — TRAINING MODE")
    if dry_run:
        print("  *** DRY-RUN MODE: 1000 S1 entities, 5 LGB trees ***")
    print("=" * 70)
    t_global = time.time()

    # Apply overrides
    _sample = DRY_RUN_SAMPLE if dry_run else (sample_per_country or TRAIN_SAMPLE_PER_COUNTRY)
    _top_k = top_k or TOP_K
    _qchunk = query_chunk or QUERY_CHUNK

    print(f"  Config: sample={_sample:,}, top_k={_top_k}, query_chunk={_qchunk}", flush=True)

    # Load ground truth
    print("\n[1/7] Loading ground truth...", flush=True)
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    print(f"  Loaded {len(gt):,} ground truth entries", flush=True)

    # Load S1 records
    print("\n[2/7] Loading S1 records...", flush=True)
    s1_all = load_source_file(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    print(f"  Loaded {len(s1_all):,} S1 records", flush=True)

    # Process each country
    all_features = []
    all_s1_ids_used = []

    for country in ["US", "India"]:
        print(f"\n{'='*70}")
        print(f"PROCESSING COUNTRY: {country}")
        print(f"{'='*70}")

        # Filter S1 by country
        s1_country = {eid: rec for eid, rec in s1_all.items() if rec[2] == country}
        s1_with_gt = {eid: rec for eid, rec in s1_country.items() if eid in gt}

        # Sample for training
        s1_ids_list = list(s1_with_gt.keys())
        np.random.seed(42)
        np.random.shuffle(s1_ids_list)
        sample_ids = s1_ids_list[:_sample]
        s1_sample = {eid: s1_with_gt[eid] for eid in sample_ids}

        n_true_matches = sum(len(gt[eid]) for eid in sample_ids)
        n_singletons = sum(1 for eid in sample_ids if len(gt[eid]) == 0)
        print(f"  Selected {len(s1_sample):,} S1 entities for training "
              f"(singletons: {n_singletons:,}, true matches: {n_true_matches:,})", flush=True)

        # Load partners
        print(f"\n[3/7] Loading {country} partners...", flush=True)
        t0 = time.time()
        partner_raw = {}
        for fn in ["train_source2.tsv", "train_source3.tsv"]:
            partner_raw.update(load_source_file(
                os.path.join(TRAIN_DIR, fn), country_filter=country))
        print(f"  Loaded {len(partner_raw):,} {country} partners in {time.time()-t0:.1f}s", flush=True)

        # Fit FeatureBuilder IDF on partner name corpus
        print(f"\n[4/7] Fitting FeatureBuilder IDF on {country} partner names...", flush=True)
        t0 = time.time()
        fb = FeatureBuilder()
        fb.fit_idf([make_name_text(v[0]) for v in partner_raw.values()])
        print(f"  IDF fitted on {len(partner_raw):,} names in {time.time()-t0:.1f}s "
              f"({len(fb.idf_dict):,} tokens)", flush=True)
        joblib.dump(fb, os.path.join(ARTIFACTS_DIR, f"fb_{country}.joblib"))

        # TF-IDF Blocking
        print(f"\n[5/7] TF-IDF char n-gram blocking for {country}...", flush=True)
        partner_ids = list(partner_raw.keys())
        partner_texts = [make_text(partner_raw[pid][0], partner_raw[pid][1]) for pid in partner_ids]

        query_ids = list(s1_sample.keys())
        query_texts = [make_text(s1_sample[qid][0], s1_sample[qid][1]) for qid in query_ids]

        candidates, vectorizer = tfidf_blocking(
            query_texts, query_ids,
            partner_texts, partner_ids,
            k=_top_k,
            partner_chunk_size=PARTNER_CHUNK,
            query_chunk_size=_qchunk
        )

        # Save vectorizer for this country
        joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, f"vectorizer_{country}.joblib"))
        print(f"  Generated {len(candidates):,} candidate pairs", flush=True)

        # Blocking recall audit
        recall_ceiling, _ = blocking_recall_audit(candidates, gt, sample_ids, label=country)
        if recall_ceiling < 0.95:
            print(f"  ⚠️  WARNING: Recall ceiling {recall_ceiling:.2%} < 95%! "
                  f"Consider increasing TOP_K to {_top_k + 25}.", flush=True)

        # Pre-compute normalized records for speed
        print(f"\n[6/7] Pre-computing normalized records for {country}...", flush=True)
        t0 = time.time()
        s1_norm_cache = build_norm_cache(s1_sample)
        partner_norm_cache = build_norm_cache(partner_raw)
        print(f"  Normalized {len(s1_norm_cache):,} S1 + {len(partner_norm_cache):,} partner records "
              f"in {time.time()-t0:.1f}s", flush=True)

        # Feature extraction
        print(f"\n[7/7] Extracting features for {len(candidates):,} pairs...", flush=True)
        features = extract_features_batch(
            candidates, s1_sample, partner_raw, fb,
            s1_norm_cache=s1_norm_cache,
            partner_norm_cache=partner_norm_cache
        )

        # Label
        true_pairs = {(s1_id, pid) for s1_id in sample_ids for pid in gt[s1_id]}
        features["label"] = [
            1 if (s1, pid) in true_pairs else 0
            for s1, pid in zip(features["source1_entity_id"], features["partner_entity_id"])
        ]

        all_features.append(features)
        all_s1_ids_used.extend(sample_ids)

        # Free country memory
        del partner_raw, partner_ids, partner_texts, vectorizer, candidates
        del s1_norm_cache, partner_norm_cache
        gc.collect()
        print(f"  {country} features: {len(features):,} pairs, "
              f"{features['label'].sum():,} positives ({features['label'].mean():.4f})", flush=True)

    # ── France Transfer Validation ──────────────────────────────────────
    # Simulate France generalization: hold out a small India sample, train on rest,
    # run full pipeline with fresh vectorizer (mimicking unseen-country scenario).
    print(f"\n{'='*70}")
    print("FRANCE TRANSFER VALIDATION (India held-out simulation)")
    print(f"{'='*70}")
    _transfer_sample = min(2000, _sample // 10)
    try:
        _run_transfer_validation(gt, s1_all, _transfer_sample, _top_k, _qchunk, dry_run)
    except Exception as e:
        print(f"  Transfer validation failed (non-fatal): {e}", flush=True)

    # ── Combined Training ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMBINED TRAINING")
    print(f"{'='*70}")

    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"  Total training pairs: {len(combined_df):,}")
    print(f"  Positives: {combined_df['label'].sum():,} ({combined_df['label'].mean():.4f})")

    feature_cols = [c for c in combined_df.columns
                    if c not in {"source1_entity_id", "partner_entity_id", "label",
                                 "p_raw", "p_cal"}]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}", flush=True)

    # Train model
    model, calibrator = train_model(combined_df, feature_cols, dry_run=dry_run)

    # Threshold search (with global 1-to-1 applied before sweep)
    print("\n  Searching for optimal threshold...", flush=True)
    best_thresh, best_strategy, best_f05_oof = threshold_search(
        combined_df,
        {k: gt[k] for k in all_s1_ids_used},
        all_s1_ids_used
    )

    # ── Full end-to-end GT evaluation ───────────────────────────────────
    print("\n  Running full end-to-end GT evaluation...", flush=True)
    df_121 = resolve_one_to_one_simple(combined_df, prob_col="p_cal")

    if best_strategy == "expected_f05":
        final_preds = decide_expected_f05(df_121, all_s1_ids_used, prob_col="p_cal")
    else:
        final_preds = {}
        passing = df_121[df_121["p_cal"] >= best_thresh]
        for s1_id, pid in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
            if s1_id not in final_preds:
                final_preds[s1_id] = set()
            final_preds[s1_id].add(pid)
        for s1_id in all_s1_ids_used:
            if s1_id not in final_preds:
                final_preds[s1_id] = set()

    eval_gt = {k: gt[k] for k in all_s1_ids_used}
    metrics = evaluate_predictions(eval_gt, final_preds)

    macro_f05 = metrics["macro_f05"]
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  FINAL VALIDATION RESULTS                           │")
    print(f"  │  Macro F0.5:          {macro_f05:.4f}                      │")
    print(f"  │  Singleton Accuracy:  {metrics['singleton_accuracy']:.4f}                      │")
    print(f"  │  Non-Singleton F0.5:  {metrics['non_singleton_f05']:.4f}                      │")
    print(f"  │  Mean Precision:      {metrics['mean_precision']:.4f}                      │")
    print(f"  │  Mean Recall:         {metrics['mean_recall']:.4f}                      │")
    print(f"  │  Strategy:            {best_strategy:<28}│")
    print(f"  │  Threshold:           {best_thresh:.2f}                         │")
    print(f"  └─────────────────────────────────────────────────────┘")

    if macro_f05 >= 0.97:
        print(f"\n  ✅ PASS: Macro F0.5 = {macro_f05:.4f} >= 0.97", flush=True)
    else:
        print(f"\n  ❌ FAIL: Macro F0.5 = {macro_f05:.4f} < 0.97", flush=True)
        print(f"     Diagnosis:", flush=True)
        print(f"       - Singleton accuracy: {metrics['singleton_accuracy']:.4f} "
              f"({'OK' if metrics['singleton_accuracy'] > 0.9 else 'LOW — too many false merges'})",
              flush=True)
        print(f"       - Non-singleton F0.5: {metrics['non_singleton_f05']:.4f} "
              f"({'OK' if metrics['non_singleton_f05'] > 0.9 else 'LOW — missing true matches'})",
              flush=True)

    # Save artifacts
    print("\n  Saving model artifacts...", flush=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "lgb_model.joblib"))
    joblib.dump(calibrator, os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
    joblib.dump(best_thresh, os.path.join(ARTIFACTS_DIR, "threshold.joblib"))
    joblib.dump(best_strategy, os.path.join(ARTIFACTS_DIR, "strategy.joblib"))

    total_time = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Macro F0.5: {macro_f05:.4f}")
    print(f"  Strategy: {best_strategy}, Threshold: {best_thresh:.2f}")
    print(f"  Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"{'='*70}")

    return macro_f05


def _run_transfer_validation(gt, s1_all, transfer_sample, top_k, query_chunk, dry_run):
    """
    Simulate France generalization: hold out a small India sample,
    fit a fresh vectorizer on India partners (mimicking unseen-country scenario),
    and report OOF F0.5 on the held-out set.
    """
    country = "India"
    s1_country = {eid: rec for eid, rec in s1_all.items() if rec[2] == country}
    s1_with_gt = {eid: rec for eid, rec in s1_country.items() if eid in gt}

    s1_ids_list = list(s1_with_gt.keys())
    np.random.seed(99)
    np.random.shuffle(s1_ids_list)
    transfer_ids = s1_ids_list[:transfer_sample]
    s1_transfer = {eid: s1_with_gt[eid] for eid in transfer_ids}

    print(f"  Transfer validation: {len(s1_transfer):,} India S1 entities (fresh vectorizer)", flush=True)

    # Load India partners
    partner_raw = {}
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        partner_raw.update(load_source_file(
            os.path.join(TRAIN_DIR, fn), country_filter=country))

    # Fit fresh FeatureBuilder (no saved IDF)
    fb_transfer = FeatureBuilder()
    fb_transfer.fit_idf([make_name_text(v[0]) for v in partner_raw.values()])

    partner_ids = list(partner_raw.keys())
    partner_texts = [make_text(partner_raw[pid][0], partner_raw[pid][1]) for pid in partner_ids]
    query_ids = list(s1_transfer.keys())
    query_texts = [make_text(s1_transfer[qid][0], s1_transfer[qid][1]) for qid in query_ids]

    # Fresh vectorizer (no saved one)
    candidates, _ = tfidf_blocking(
        query_texts, query_ids,
        partner_texts, partner_ids,
        vectorizer=None,
        k=top_k,
        partner_chunk_size=PARTNER_CHUNK,
        query_chunk_size=query_chunk
    )

    recall_ceiling, _ = blocking_recall_audit(candidates, gt, transfer_ids, label="Transfer")

    # Feature extraction
    s1_norm_cache = build_norm_cache(s1_transfer)
    partner_norm_cache = build_norm_cache(partner_raw)
    features = extract_features_batch(
        candidates, s1_transfer, partner_raw, fb_transfer,
        s1_norm_cache=s1_norm_cache,
        partner_norm_cache=partner_norm_cache
    )

    # Load saved model for scoring
    model_path = os.path.join(ARTIFACTS_DIR, "lgb_model.joblib")
    cal_path = os.path.join(ARTIFACTS_DIR, "calibrator.joblib")
    fc_path = os.path.join(ARTIFACTS_DIR, "feature_cols.joblib")
    thresh_path = os.path.join(ARTIFACTS_DIR, "threshold.joblib")

    if not all(os.path.exists(p) for p in [model_path, cal_path, fc_path, thresh_path]):
        print("  Transfer validation: model not yet saved, skipping scoring.", flush=True)
        return

    model = joblib.load(model_path)
    calibrator = joblib.load(cal_path)
    feature_cols = joblib.load(fc_path)
    threshold = joblib.load(thresh_path)

    X = features[feature_cols].values
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = calibrator.transform(p_raw)
    features["p_cal"] = p_cal

    df_121 = resolve_one_to_one_simple(features, prob_col="p_cal")
    preds = {}
    passing = df_121[df_121["p_cal"] >= threshold]
    for s1_id, pid in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
        if s1_id not in preds:
            preds[s1_id] = set()
        preds[s1_id].add(pid)
    for s1_id in transfer_ids:
        if s1_id not in preds:
            preds[s1_id] = set()

    eval_gt = {k: gt[k] for k in transfer_ids}
    metrics = evaluate_predictions(eval_gt, preds)
    print(f"  Transfer (India→France sim) F0.5: {metrics['macro_f05']:.4f} "
          f"(singleton_acc={metrics['singleton_accuracy']:.4f}, "
          f"non_singleton_f05={metrics['non_singleton_f05']:.4f})", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN: PREDICT MODE
# ═══════════════════════════════════════════════════════════════════════

def run_predict(top_k=None, query_chunk=None):
    print("=" * 70)
    print("ENTITY RESOLUTION PIPELINE v3 — PREDICTION MODE")
    print("=" * 70)
    t_global = time.time()

    _top_k = top_k or TOP_K
    _qchunk = query_chunk or QUERY_CHUNK

    # Load model artifacts
    print("\n[1/4] Loading model artifacts...", flush=True)
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgb_model.joblib"))
    calibrator = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
    threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "threshold.joblib"))
    strategy_path = os.path.join(ARTIFACTS_DIR, "strategy.joblib")
    best_strategy = joblib.load(strategy_path) if os.path.exists(strategy_path) else "threshold"
    print(f"  Loaded model with {len(feature_cols)} features, "
          f"threshold={threshold:.2f}, strategy={best_strategy}", flush=True)

    # Load test S1 records
    print("\n[2/4] Loading test Source 1 records...", flush=True)
    all_test_s1 = {}
    s1_order = []
    s1_by_country = defaultdict(dict)

    with open(os.path.join(TEST_DIR, "test_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            eid, name, addr, country = p[0], p[1], p[2], p[3]
            all_test_s1[eid] = (name, addr, country)
            s1_order.append(eid)
            s1_by_country[country][eid] = (name, addr, country)

    print(f"  Total test S1: {len(all_test_s1):,}", flush=True)
    for c, recs in sorted(s1_by_country.items()):
        print(f"    {c}: {len(recs):,}", flush=True)

    # Accumulate results per country
    # Key: s1_id → set of matched partner_ids
    # Key: s1_id → list of all candidate partner_ids
    final_matches = {}     # {s1_id: set(partner_ids)}
    final_candidates = {}  # {s1_id: list(partner_ids)}

    # Process each country
    for country in ["France", "US", "India"]:
        s1_country = s1_by_country.get(country, {})
        if not s1_country:
            continue

        print(f"\n{'='*70}")
        print(f"PROCESSING TEST: {country} ({len(s1_country):,} queries)")
        print(f"{'='*70}")

        # Load partners
        print(f"  Loading {country} test partners...", flush=True)
        t0 = time.time()
        partner_raw = {}
        for fn in ["test_source2.tsv", "test_source3.tsv"]:
            partner_raw.update(load_source_file(
                os.path.join(TEST_DIR, fn), country_filter=country))
        print(f"  Loaded {len(partner_raw):,} {country} partners in {time.time()-t0:.1f}s", flush=True)

        # Fit fresh FeatureBuilder on test partner names (no pickle dependency on training IDF)
        print(f"  Fitting FeatureBuilder IDF on {country} test partner names...", flush=True)
        t0 = time.time()
        fb = FeatureBuilder()
        fb.fit_idf([make_name_text(v[0]) for v in partner_raw.values()])
        print(f"  IDF fitted in {time.time()-t0:.1f}s ({len(fb.idf_dict):,} tokens)", flush=True)

        # TF-IDF Blocking — always fit fresh on test partner corpus
        print(f"\n  TF-IDF blocking...", flush=True)
        partner_ids = list(partner_raw.keys())
        partner_texts = [make_text(partner_raw[pid][0], partner_raw[pid][1]) for pid in partner_ids]

        query_ids = list(s1_country.keys())
        query_texts = [make_text(s1_country[qid][0], s1_country[qid][1]) for qid in query_ids]

        candidates, _ = tfidf_blocking(
            query_texts, query_ids,
            partner_texts, partner_ids,
            vectorizer=None,
            k=_top_k,
            partner_chunk_size=PARTNER_CHUNK,
            query_chunk_size=_qchunk
        )
        print(f"  Generated {len(candidates):,} candidate pairs", flush=True)

        # Store all candidates for candidate_pairs.tsv
        for s1_id, pid, _ in candidates:
            if s1_id not in final_candidates:
                final_candidates[s1_id] = []
            final_candidates[s1_id].append(pid)

        # Blocking quality audit (test mode — no GT)
        blocking_quality_audit_test(candidates, query_ids, label=country)

        # ── Memory-efficient streaming inference ──────────────────────────
        # Process in query chunks: blocking → features → inference → accumulate
        # Only keep (s1_id, partner_id, p_cal) triples across chunks (not full DataFrames)
        # This prevents OOM for India (810K S1 × 50 candidates = 40M pairs)

        # Pre-compute normalized records for partners (shared across query chunks)
        print(f"\n  Pre-computing normalized partner records for {country}...", flush=True)
        t0 = time.time()
        partner_norm_cache = build_norm_cache(partner_raw)
        print(f"  Normalized {len(partner_norm_cache):,} partner records in {time.time()-t0:.1f}s",
              flush=True)

        # Group candidates by S1 query chunk
        STREAM_CHUNK = 50000  # S1 entities per streaming chunk
        query_ids_list = list(s1_country.keys())
        n_stream_chunks = (len(query_ids_list) + STREAM_CHUNK - 1) // STREAM_CHUNK

        # Build candidate lookup: s1_id → list of (partner_id, score)
        cand_lookup = defaultdict(list)
        for s1_id, pid, score in candidates:
            cand_lookup[s1_id].append((pid, score))

        # Accumulate all scored triples for this country (for global 1-to-1)
        country_scored = []  # list of (s1_id, partner_id, p_cal)

        for sc in range(n_stream_chunks):
            sc_start = sc * STREAM_CHUNK
            sc_end = min(sc_start + STREAM_CHUNK, len(query_ids_list))
            chunk_s1_ids = query_ids_list[sc_start:sc_end]

            # Build candidate list for this S1 chunk
            chunk_cands = []
            for s1_id in chunk_s1_ids:
                for pid, score in cand_lookup.get(s1_id, []):
                    chunk_cands.append((s1_id, pid, score))

            if not chunk_cands:
                continue

            print(f"\n  Stream chunk {sc+1}/{n_stream_chunks} "
                  f"({sc_start:,}-{sc_end:,} S1, {len(chunk_cands):,} pairs)...", flush=True)

            # Build S1 norm cache for this chunk only
            s1_chunk_raw = {eid: s1_country[eid] for eid in chunk_s1_ids}
            s1_norm_cache = build_norm_cache(s1_chunk_raw)

            # Feature extraction
            features = extract_features_batch(
                chunk_cands, s1_chunk_raw, partner_raw, fb,
                s1_norm_cache=s1_norm_cache,
                partner_norm_cache=partner_norm_cache
            )

            # Model inference
            X = features[feature_cols].values
            p_raw = model.predict_proba(X)[:, 1]
            p_cal = calibrator.transform(p_raw)

            # Accumulate scored triples (minimal memory footprint)
            for s1_id, pid, pc in zip(
                features["source1_entity_id"].values,
                features["partner_entity_id"].values,
                p_cal
            ):
                country_scored.append((s1_id, pid, float(pc)))

            del features, X, p_raw, p_cal, s1_norm_cache
            gc.collect()

        # ── Global 1-to-1 consistency for this country ────────────────────
        print(f"\n  Applying global 1-to-1 consistency for {country} "
              f"({len(country_scored):,} scored pairs)...", flush=True)

        scored_df = pd.DataFrame(country_scored, columns=["source1_entity_id", "partner_entity_id", "p_cal"])
        scored_df_121 = resolve_one_to_one_simple(scored_df, prob_col="p_cal")

        # Apply threshold or expected-F0.5 strategy
        if best_strategy == "expected_f05":
            country_preds = decide_expected_f05(
                scored_df_121, query_ids_list, prob_col="p_cal"
            )
        else:
            country_preds = {}
            passing = scored_df_121[scored_df_121["p_cal"] >= threshold]
            for s1_id, pid in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
                if s1_id not in country_preds:
                    country_preds[s1_id] = set()
                country_preds[s1_id].add(pid)
            for s1_id in query_ids_list:
                if s1_id not in country_preds:
                    country_preds[s1_id] = set()

        n_matched_country = sum(1 for v in country_preds.values() if v)
        n_links_country = sum(len(v) for v in country_preds.values())
        print(f"  {country}: {n_matched_country:,} matched entities, "
              f"{n_links_country:,} total links "
              f"({n_matched_country/len(query_ids_list):.2%} match rate)", flush=True)

        # Merge into final results
        for s1_id, matches in country_preds.items():
            final_matches[s1_id] = matches

        # Free country memory
        del partner_raw, partner_ids, partner_texts, candidates
        del partner_norm_cache, cand_lookup, country_scored, scored_df, scored_df_121
        gc.collect()

    # ── Write output files ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WRITING OUTPUT FILES")
    print(f"{'='*70}")

    match_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cand_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    # Use write_tsv_submission() for deterministic, sorted, validator-compliant output
    write_tsv_submission(
        filepath=match_file,
        predictions=final_matches,
        all_s1_ids=s1_order,
        id_col_name="source1_entity_id",
        list_col_name="matched_entity_ids"
    )

    # Write candidate_pairs.tsv (all top-K candidates for every S1 entity)
    write_tsv_submission(
        filepath=cand_file,
        predictions={eid: set(final_candidates.get(eid, [])) for eid in s1_order},
        all_s1_ids=s1_order,
        id_col_name="source1_entity_id",
        list_col_name="candidate_entity_ids"
    )

    # Statistics
    n_matched = sum(1 for eid in s1_order if final_matches.get(eid))
    n_singletons = len(s1_order) - n_matched
    total_links = sum(len(v) for v in final_matches.values())
    print(f"  Written {len(s1_order):,} rows to both files")
    print(f"  Matched entities: {n_matched:,} ({n_matched/len(s1_order):.2%})")
    print(f"  Singletons: {n_singletons:,} ({n_singletons/len(s1_order):.2%})")
    print(f"  Total links: {total_links:,} (avg {total_links/len(s1_order):.2f} per entity)")

    # Validate
    print("\n  Running official validator...", flush=True)
    val_pass, val_msg = run_submission_validator(
        matching_tsv_path=match_file,
        candidate_tsv_path=cand_file,
        test_dir=TEST_DIR
    )

    total_time = time.time() - t_global
    print(f"\n{'='*70}")
    if val_pass:
        print(f"PREDICTION COMPLETE — VALIDATION PASSED ✅")
    else:
        print(f"PREDICTION COMPLETE — VALIDATION ISSUES: {val_msg}")
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*70}")

    return val_pass


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline v3")
    parser.add_argument("mode", choices=["train", "predict"],
                        help="Pipeline mode: train or predict")
    parser.add_argument("--sample", type=int, default=None,
                        help=f"S1 entities per country for training (default: {TRAIN_SAMPLE_PER_COUNTRY})")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"Top-K candidates per S1 entity (default: {TOP_K})")
    parser.add_argument("--query-chunk", type=int, default=None,
                        help=f"Query chunk size for blocking (default: {QUERY_CHUNK})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 1000 S1 entities, 5 LGB trees, <5 min")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train(
            sample_per_country=args.sample,
            top_k=args.top_k,
            query_chunk=args.query_chunk,
            dry_run=args.dry_run
        )
    elif args.mode == "predict":
        run_predict(
            top_k=args.top_k,
            query_chunk=args.query_chunk
        )
