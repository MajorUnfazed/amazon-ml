"""
Entity Resolution Pipeline v3 — PRODUCTION
============================================
Hybrid approach: Fast inverted-index blocking + rich features + LightGBM.

Key fixes vs. v2:
  - Posting list cap: 2% of corpus (not hardcoded 1500)
  - Multi-channel: word IDF + char 3-grams + slug + address keys + postal
  - Full normalize_record() features (name_core, legal suffix, acronyms, etc.)
  - Train on both US AND India (not just 10K US)
  - All features computed at inference time (no pickle version issues)
  - Global 1-to-1 consistency per country (not per chunk)
  - TF-IDF cosine similarity computed ON retrieved candidates (fast & accurate)

Usage:
  python pipeline_v3_fast.py train [--sample N]
  python pipeline_v3_fast.py predict
"""

import os, sys, time, gc, math, re, unicodedata, argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from rapidfuzz import fuzz, distance
import joblib
import heapq

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
from decide.one_to_one import resolve_one_to_one
from pipeline_io.writer import write_tsv_submission, run_submission_validator

# Blocking config
TOP_K = 50                # candidates per S1 entity
MAX_DF_FRAC = 0.03        # skip words in >3% of corpus (IDF too low to help)
MIN_POSTING = 5000        # minimum posting list cap (even for small corpora)
CHAR3_MAX_DF_FRAC = 0.02  # skip char 3-grams in >2% of corpus

# Training config
TRAIN_SAMPLE_PER_COUNTRY = 50000
CHUNK_SIZE = 50000  # S1 queries per processing chunk

# LightGBM params
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


# ─── Text Utilities ────────────────────────────────────────────────────
def make_text(name, addr):
    """Combined text for TF-IDF."""
    text = str(name) + " " + str(addr)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Data Loading ──────────────────────────────────────────────────────
def load_source_file(filepath, country_filter=None):
    records = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                if country_filter and p[3] != country_filter:
                    continue
                records[p[0]] = (p[1], p[2], p[3])
    return records

def load_ground_truth(filepath):
    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            matches = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt[eid] = matches
    return gt


# ─── Multi-Channel Inverted Index Blocking ─────────────────────────────
def build_index_and_retrieve(s1_records, partner_records, country, top_k=50):
    """
    Build multi-channel inverted index on partner_records and retrieve
    top-K candidates for each S1 record.
    
    Key fix: posting list cap = max(N * MAX_DF_FRAC, MIN_POSTING).
    
    s1_records: dict of {eid: normalized_record_dict}
    partner_records: dict of {pid: normalized_record_dict}
    
    Returns: list of (s1_id, partner_id, retriever_score)
    """
    N = len(partner_records)
    max_posting = max(int(N * MAX_DF_FRAC), MIN_POSTING)
    max_posting_c3 = max(int(N * CHAR3_MAX_DF_FRAC), MIN_POSTING)
    
    print(f"    Building index: {N:,} partners, posting cap={max_posting:,} (words), "
          f"{max_posting_c3:,} (char3)", flush=True)
    t0 = time.time()
    
    # Count document frequencies first
    word_df = Counter()
    char3_df = Counter()
    
    for pid, norm in partner_records.items():
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
    
    # Compute IDF
    word_idf = {}
    for w, df in word_df.items():
        if df <= max_posting:
            word_idf[w] = math.log(1.0 + N / df)
    
    char3_idf = {}
    for c3, df in char3_df.items():
        if df <= max_posting_c3:
            char3_idf[c3] = math.log(1.0 + N / df)
    
    print(f"    IDF computed: {len(word_idf):,} words, {len(char3_idf):,} char3 "
          f"(dropped {len(word_df) - len(word_idf):,} / {len(char3_df) - len(char3_idf):,})", flush=True)
    
    # Build inverted indices
    slug_index = defaultdict(list)
    word_index = defaultdict(list)
    char3_index = defaultdict(list)
    addr_key_index = defaultdict(list)
    postal_index = defaultdict(list)
    
    for pid, norm in partner_records.items():
        n_clean = str(norm["name_clean"])
        n_core = str(norm["name_core"])
        a_clean = str(norm["address_clean"])
        
        # Channel 1: Core name slug
        slug = n_core.replace(" ", "")[:25]
        if len(slug) >= 4:
            slug_index[slug].append(pid)
        
        # Channel 2: Address key (building number + first street word)
        nums = norm.get("building_numbers", [])
        a_toks = [t for t in a_clean.split() if len(t) >= 3 and not t.isdigit()]
        if nums and a_toks:
            addr_key_index[f"{nums[0]}_{a_toks[0]}"].append(pid)
        
        # Channel 3: Word index (name + address, IDF-filtered)
        for w in set(n_clean.split()):
            if w in word_idf:
                word_index[w].append(pid)
        for w in set(a_clean.split()):
            if w in word_idf and len(w) >= 4 and not w.isdigit():
                word_index[w].append(pid)
        
        # Channel 4: Char 3-grams
        core_flat = n_core.replace(" ", "")
        for c3 in {core_flat[i:i+3] for i in range(len(core_flat) - 2)}:
            if c3 in char3_idf:
                char3_index[c3].append(pid)
        
        # Channel 5: Postal code
        pc = norm.get("postal_code")
        if pc:
            postal_index[pc].append(pid)
    
    print(f"    Index built in {time.time()-t0:.1f}s", flush=True)
    
    # Retrieve candidates for each S1
    print(f"    Retrieving top-{top_k} candidates for {len(s1_records):,} S1 entities...", flush=True)
    t0 = time.time()
    all_candidates = []
    
    for s1_id, norm in s1_records.items():
        cand_scores = defaultdict(float)
        n_clean = str(norm["name_clean"])
        n_core = str(norm["name_core"])
        a_clean = str(norm["address_clean"])
        
        # 1. Exact core slug (high weight)
        slug = n_core.replace(" ", "")[:25]
        if len(slug) >= 4:
            for pid in slug_index.get(slug, []):
                cand_scores[pid] += 30.0
        
        # 2. Address key (building + street)
        nums = norm.get("building_numbers", [])
        a_toks = [t for t in a_clean.split() if len(t) >= 3 and not t.isdigit()]
        if nums and a_toks:
            addr_k = f"{nums[0]}_{a_toks[0]}"
            pids = addr_key_index.get(addr_k, [])
            if 0 < len(pids) <= max_posting:
                for pid in pids:
                    cand_scores[pid] += 25.0
        
        # 3. Word tokens (name + address, IDF-weighted)
        for w in set(n_clean.split()):
            idf = word_idf.get(w, 0.0)
            if idf > 0:
                for pid in word_index.get(w, []):
                    cand_scores[pid] += idf
        
        for w in set(a_clean.split()):
            if len(w) >= 4 and not w.isdigit():
                idf = word_idf.get(w, 0.0)
                if idf > 0:
                    for pid in word_index.get(w, []):
                        cand_scores[pid] += idf * 0.7
        
        # 4. Char 3-grams (name core)
        core_flat = n_core.replace(" ", "")
        for c3 in {core_flat[i:i+3] for i in range(len(core_flat) - 2)}:
            idf = char3_idf.get(c3, 0.0)
            if idf > 0:
                for pid in char3_index.get(c3, []):
                    cand_scores[pid] += idf * 0.4
        
        # 5. Postal code bonus
        pc = norm.get("postal_code")
        if pc:
            for pid in postal_index.get(pc, []):
                if pid in cand_scores:
                    cand_scores[pid] += 5.0
        
        # Extract top-K
        if cand_scores:
            top_cands = heapq.nlargest(top_k, cand_scores.items(), key=lambda x: x[1])
            for pid, score in top_cands:
                all_candidates.append((s1_id, pid, float(score)))
    
    elapsed = time.time() - t0
    avg_cands = len(all_candidates) / len(s1_records) if s1_records else 0
    print(f"    Retrieved {len(all_candidates):,} candidates "
          f"(avg {avg_cands:.1f}/entity) in {elapsed:.1f}s", flush=True)
    
    return all_candidates


# ─── Feature Engineering ───────────────────────────────────────────────
def compute_tfidf_cosine_batch(candidates, s1_records, partner_records):
    """Compute TF-IDF char n-gram cosine similarity for candidate pairs using scipy sparse matmul.
    Only processes the unique S1s and partners that appear in candidates (not all records)."""
    
    # Collect unique IDs
    s1_ids_in_cands = list(set(c[0] for c in candidates))
    p_ids_in_cands = list(set(c[1] for c in candidates))
    
    if not s1_ids_in_cands or not p_ids_in_cands:
        return {}
    
    # Prepare texts
    s1_texts = [make_text(s1_records[eid]["name_clean"], s1_records[eid]["address_clean"]) 
                for eid in s1_ids_in_cands]
    p_texts = [make_text(partner_records[pid]["name_clean"], partner_records[pid]["address_clean"])
               for pid in p_ids_in_cands]
    
    # Fit TF-IDF on partner texts (small set, fast)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), 
                          max_features=50000, sublinear_tf=True, dtype=np.float32)
    P_mat = vec.fit_transform(p_texts)
    Q_mat = vec.transform(s1_texts)
    
    # Build lookup indices
    s1_idx = {eid: i for i, eid in enumerate(s1_ids_in_cands)}
    p_idx = {pid: i for i, pid in enumerate(p_ids_in_cands)}
    
    # Compute cosine similarities only for candidate pairs
    sim_matrix = Q_mat.dot(P_mat.T).tocsr()
    
    cosine_scores = {}
    for s1_id, pid, _ in candidates:
        i = s1_idx[s1_id]
        j = p_idx[pid]
        cosine_scores[(s1_id, pid)] = float(sim_matrix[i, j])
    
    return cosine_scores


def extract_features_batch(candidates, s1_norm, partner_norm, fb, cosine_scores=None):
    """Extract ~40 features per candidate pair."""
    rows = []
    n_total = len(candidates)
    t0 = time.time()
    
    for idx, (s1_id, p_id, retriever_score) in enumerate(candidates):
        s1n = s1_norm.get(s1_id, {})
        pn = partner_norm.get(p_id, {})
        
        s1_clean = str(s1n.get("name_clean", ""))
        s1_core = str(s1n.get("name_core", ""))
        s1_suffix = s1n.get("legal_suffix", "")
        s1_acr = s1n.get("name_acronym", "")
        s1_vars = s1n.get("name_variants", [s1_clean])
        s1_addr = str(s1n.get("address_clean", ""))
        s1_postal = s1n.get("postal_code", "")
        s1_nums = set(s1n.get("building_numbers", []))
        s1_lms = set(s1n.get("landmarks", []))
        
        p_clean = str(pn.get("name_clean", ""))
        p_core = str(pn.get("name_core", ""))
        p_suffix = pn.get("legal_suffix", "")
        p_acr = pn.get("name_acronym", "")
        p_vars = pn.get("name_variants", [p_clean])
        p_addr = str(pn.get("address_clean", ""))
        p_postal = pn.get("postal_code", "")
        p_nums = set(pn.get("building_numbers", []))
        p_lms = set(pn.get("landmarks", []))
        
        feat = {"source1_entity_id": s1_id, "partner_entity_id": p_id}
        
        # 1. Retriever score
        feat["retriever_score"] = retriever_score
        
        # 2. TF-IDF cosine (computed separately for efficiency)
        feat["tfidf_cosine"] = cosine_scores.get((s1_id, p_id), 0.0) if cosine_scores else 0.0
        
        # 3. Name similarities (clean)
        feat["name_clean_jw"] = float(distance.JaroWinkler.similarity(s1_clean, p_clean))
        feat["name_clean_lev"] = float(fuzz.ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_tsort"] = float(fuzz.token_sort_ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_tset"] = float(fuzz.token_set_ratio(s1_clean, p_clean) / 100.0)
        feat["name_clean_partial"] = float(fuzz.partial_ratio(s1_clean, p_clean) / 100.0)
        
        # 4. Name similarities (core)
        feat["name_core_jw"] = float(distance.JaroWinkler.similarity(s1_core, p_core))
        feat["name_core_lev"] = float(fuzz.ratio(s1_core, p_core) / 100.0)
        feat["name_core_tsort"] = float(fuzz.token_sort_ratio(s1_core, p_core) / 100.0)
        feat["name_core_tset"] = float(fuzz.token_set_ratio(s1_core, p_core) / 100.0)
        
        # 5. DBA variant best match
        best_dba = max((fuzz.ratio(v1, v2) / 100.0 for v1 in s1_vars for v2 in p_vars), default=0.0)
        feat["name_dba_best"] = float(best_dba)
        
        # 6. Acronym & suffix
        feat["acronym_match"] = 1.0 if (s1_acr and p_acr and s1_acr == p_acr) else 0.0
        if s1_suffix and p_suffix:
            feat["suffix_status"] = 1.0 if s1_suffix == p_suffix else -1.0
        else:
            feat["suffix_status"] = 0.0
        
        # 7. IDF-weighted token features
        s1_toks = set(s1_core.split())
        p_toks = set(p_core.split())
        shared = s1_toks & p_toks
        unshared = (s1_toks | p_toks) - shared
        
        shared_idfs = [fb.get_token_idf(t) for t in shared]
        unshared_idfs = [fb.get_token_idf(t) for t in unshared]
        
        feat["token_shared_count"] = float(len(shared))
        feat["token_shared_idf_sum"] = float(sum(shared_idfs))
        feat["token_shared_max_idf"] = float(max(shared_idfs)) if shared_idfs else 0.0
        feat["token_unshared_max_idf"] = float(max(unshared_idfs)) if unshared_idfs else 0.0
        feat["token_jaccard"] = float(len(shared) / len(s1_toks | p_toks)) if (s1_toks | p_toks) else 0.0
        
        # 8. Address similarities
        feat["addr_tset"] = float(fuzz.token_set_ratio(s1_addr, p_addr) / 100.0)
        feat["addr_tsort"] = float(fuzz.token_sort_ratio(s1_addr, p_addr) / 100.0)
        feat["addr_lev"] = float(fuzz.ratio(s1_addr, p_addr) / 100.0)
        feat["addr_partial"] = float(fuzz.partial_ratio(s1_addr, p_addr) / 100.0)
        
        s1_a_toks = set(s1_addr.split())
        p_a_toks = set(p_addr.split())
        shared_a = s1_a_toks & p_a_toks
        feat["addr_jaccard"] = float(len(shared_a) / len(s1_a_toks | p_a_toks)) if (s1_a_toks | p_a_toks) else 0.0
        
        # 9. Postal code
        if s1_postal and p_postal:
            feat["postal_match"] = 1.0 if s1_postal == p_postal else -1.0
            feat["postal_prefix3"] = 1.0 if s1_postal[:3] == p_postal[:3] else -1.0
        else:
            feat["postal_match"] = 0.0
            feat["postal_prefix3"] = 0.0
        
        # 10. Building number
        if s1_nums and p_nums:
            feat["building_match"] = 1.0 if (s1_nums & p_nums) else -1.0
        else:
            feat["building_match"] = 0.0
        
        # 11. Landmark
        feat["landmark_shared"] = 1.0 if (s1_lms and p_lms and (s1_lms & p_lms)) else 0.0
        
        # 12. Length features
        ml = max(len(s1_core), len(p_core))
        feat["name_len_ratio"] = float(min(len(s1_core), len(p_core)) / ml) if ml > 0 else 1.0
        feat["name_len_diff"] = float(abs(len(s1_core) - len(p_core)))
        mla = max(len(s1_addr), len(p_addr))
        feat["addr_len_ratio"] = float(min(len(s1_addr), len(p_addr)) / mla) if mla > 0 else 1.0
        
        # 13. Source indicator
        feat["is_source2"] = 1.0 if str(p_id).startswith("S2-") else 0.0
        
        rows.append(feat)
        
        if (idx + 1) % 500000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate
            print(f"      Features: {idx+1:,}/{n_total:,} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    
    df = pd.DataFrame(rows)
    
    # Context/rank features
    df = df.sort_values(["source1_entity_id", "retriever_score"], ascending=[True, False])
    df["rank_within_s1"] = df.groupby("source1_entity_id").cumcount() + 1.0
    best = df.groupby("source1_entity_id")["retriever_score"].transform("first")
    df["gap_to_best"] = best - df["retriever_score"]
    df["num_candidates"] = df.groupby("source1_entity_id")["partner_entity_id"].transform("count").astype(float)
    
    # Reverse rank
    df = df.sort_values(["partner_entity_id", "retriever_score"], ascending=[True, False])
    df["reverse_rank"] = df.groupby("partner_entity_id").cumcount() + 1.0
    df["num_claims"] = df.groupby("partner_entity_id")["source1_entity_id"].transform("count").astype(float)
    df["is_mutual_best"] = ((df["rank_within_s1"] == 1.0) & (df["reverse_rank"] == 1.0)).astype(float)
    
    df = df.sort_values(["source1_entity_id", "retriever_score"], ascending=[True, False]).reset_index(drop=True)
    
    print(f"      Features done: {len(df):,} pairs in {time.time()-t0:.1f}s", flush=True)
    return df


# ─── Normalize all records ─────────────────────────────────────────────
def normalize_all(raw_dict, label="records"):
    """Batch normalize_record for all entries."""
    t0 = time.time()
    norm = {}
    n = len(raw_dict)
    for i, (eid, (name, addr, country)) in enumerate(raw_dict.items()):
        norm[eid] = normalize_record(name, addr, country)
        if (i + 1) % 500000 == 0:
            print(f"      Normalized {i+1:,}/{n:,} {label}...", flush=True)
    print(f"      Normalized {n:,} {label} in {time.time()-t0:.1f}s", flush=True)
    return norm


# ═══════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════
def run_train(sample_per_country=None):
    print("=" * 70)
    print("PIPELINE v3 FAST — TRAINING")
    print("=" * 70)
    t_global = time.time()
    
    _sample = sample_per_country or TRAIN_SAMPLE_PER_COUNTRY
    print(f"  Sample per country: {_sample:,}", flush=True)
    
    # Load ground truth
    print("\n[1] Loading ground truth...", flush=True)
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    print(f"  {len(gt):,} entries", flush=True)
    
    # Load S1
    print("\n[2] Loading S1 records...", flush=True)
    s1_all = load_source_file(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    print(f"  {len(s1_all):,} S1 records", flush=True)
    
    all_features = []
    all_s1_ids = []
    
    for country in ["US", "India"]:
        print(f"\n{'='*70}\n  COUNTRY: {country}\n{'='*70}", flush=True)
        
        # Filter & sample
        s1_country = {eid: rec for eid, rec in s1_all.items() if rec[2] == country and eid in gt}
        ids = list(s1_country.keys())
        np.random.seed(42)
        np.random.shuffle(ids)
        sample_ids = ids[:_sample]
        s1_sample = {eid: s1_country[eid] for eid in sample_ids}
        
        n_true = sum(len(gt[eid]) for eid in sample_ids)
        n_sing = sum(1 for eid in sample_ids if len(gt[eid]) == 0)
        print(f"  Sample: {len(s1_sample):,} S1, {n_sing:,} singletons, {n_true:,} true matches", flush=True)
        
        # Load partners
        print(f"\n  Loading {country} partners...", flush=True)
        partner_raw = {}
        for fn in ["train_source2.tsv", "train_source3.tsv"]:
            partner_raw.update(load_source_file(os.path.join(TRAIN_DIR, fn), country_filter=country))
        print(f"  {len(partner_raw):,} partners loaded", flush=True)
        
        # Normalize ALL
        print(f"\n  Normalizing records...", flush=True)
        s1_norm = normalize_all(s1_sample, "S1")
        partner_norm = normalize_all(partner_raw, "partners")
        
        # Fit FeatureBuilder IDF
        fb = FeatureBuilder()
        fb.fit_idf([str(n.get("name_clean", "")) for n in partner_norm.values()])
        joblib.dump(fb, os.path.join(ARTIFACTS_DIR, f"fb_{country}.joblib"))
        print(f"  FeatureBuilder IDF: {len(fb.idf_dict):,} tokens", flush=True)
        
        # Blocking
        print(f"\n  Blocking...", flush=True)
        candidates = build_index_and_retrieve(s1_norm, partner_norm, country, top_k=TOP_K)
        
        # Blocking recall audit
        cand_dict = defaultdict(set)
        for s1_id, pid, _ in candidates:
            cand_dict[s1_id].add(pid)
        found = sum(1 for eid in sample_ids for pid in gt[eid] if pid in cand_dict.get(eid, set()))
        total = sum(len(gt[eid]) for eid in sample_ids)
        recall = found / total if total > 0 else 0
        print(f"  Blocking recall: {recall:.4f} ({found:,}/{total:,})", flush=True)
        
        # TF-IDF cosine on candidates
        print(f"\n  Computing TF-IDF cosine for {len(candidates):,} pairs...", flush=True)
        t0 = time.time()
        cosine_scores = compute_tfidf_cosine_batch(candidates, s1_norm, partner_norm)
        print(f"  TF-IDF cosine computed in {time.time()-t0:.1f}s", flush=True)
        
        # Features
        print(f"\n  Extracting features...", flush=True)
        features = extract_features_batch(candidates, s1_norm, partner_norm, fb, cosine_scores)
        
        # Label
        true_pairs = {(s1, pid) for s1 in sample_ids for pid in gt[s1]}
        features["label"] = [1 if (s1, pid) in true_pairs else 0
                             for s1, pid in zip(features["source1_entity_id"], features["partner_entity_id"])]
        
        all_features.append(features)
        all_s1_ids.extend(sample_ids)
        
        print(f"  {country}: {len(features):,} pairs, {features['label'].sum():,} positives "
              f"({features['label'].mean():.4f})", flush=True)
        
        del partner_raw, partner_norm, s1_norm, candidates, cosine_scores
        gc.collect()
    
    # Combined training
    print(f"\n{'='*70}\n  TRAINING\n{'='*70}", flush=True)
    combined = pd.concat(all_features, ignore_index=True)
    feature_cols = [c for c in combined.columns 
                    if c not in {"source1_entity_id", "partner_entity_id", "label", "p_raw", "p_cal"}]
    
    print(f"  Total pairs: {len(combined):,}, positives: {combined['label'].sum():,}", flush=True)
    print(f"  Features ({len(feature_cols)}): {feature_cols}", flush=True)
    
    # Train LightGBM
    X = combined[feature_cols].values
    y = combined["label"].values.astype(int)
    groups = combined["source1_entity_id"].values
    
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = min(n_neg / n_pos, 20.0) if n_pos > 0 else 1.0
    
    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = spw
    
    print(f"\n  Training 5-fold GroupKFold (spw={spw:.1f})...", flush=True)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(combined), dtype=float)
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        print(f"    Fold {fold}: {clf.best_iteration_} iters", flush=True)
    
    # Calibrate
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y)
    combined["p_cal"] = cal.transform(oof)
    
    # Final model
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    
    # Feature importances
    imp = sorted(zip(feature_cols, final_model.feature_importances_), key=lambda x: -x[1])
    print("\n  Top 15 features:", flush=True)
    for i, (fn, v) in enumerate(imp[:15], 1):
        print(f"    {i:2d}. {fn:<30}: {v:6.0f}", flush=True)
    
    # Threshold search
    print("\n  Threshold search...", flush=True)
    eval_gt = {k: gt[k] for k in all_s1_ids}
    
    # Apply 1-to-1 first
    df_121 = resolve_one_to_one(combined, prob_col="p_cal")
    
    best_f05 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.10, 0.90, 0.02):
        preds = defaultdict(set)
        for s1, pid in zip(df_121.loc[df_121["p_cal"] >= thresh, "source1_entity_id"],
                           df_121.loc[df_121["p_cal"] >= thresh, "partner_entity_id"]):
            preds[s1].add(pid)
        for eid in all_s1_ids:
            if eid not in preds:
                preds[eid] = set()
        
        scores = [((1.25 * len(eval_gt[s1] & preds[s1])) / (0.25 * len(eval_gt[s1]) + len(preds[s1]))
                   if eval_gt[s1] and preds[s1] else
                   (1.0 if not eval_gt[s1] and not preds[s1] else 0.0))
                  for s1 in all_s1_ids]
        f05 = np.mean(scores)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)
        print(f"    {thresh:.2f}: F0.5 = {f05:.4f}", flush=True)
    
    print(f"\n  Best threshold: {best_thresh:.2f} (F0.5 = {best_f05:.4f})", flush=True)
    
    # Save
    joblib.dump(final_model, os.path.join(ARTIFACTS_DIR, "lgb_model.joblib"))
    joblib.dump(cal, os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
    joblib.dump(best_thresh, os.path.join(ARTIFACTS_DIR, "threshold.joblib"))
    
    total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE in {total:.0f}s ({total/3600:.1f}h)")
    print(f"  F0.5 = {best_f05:.4f}, threshold = {best_thresh:.2f}")
    print(f"{'='*70}")
    return best_f05


# ═══════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════
def run_predict():
    print("=" * 70)
    print("PIPELINE v3 FAST — PREDICTION")
    print("=" * 70)
    t_global = time.time()
    
    # Load artifacts
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgb_model.joblib"))
    cal = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))
    threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "threshold.joblib"))
    print(f"  Model: {len(feature_cols)} features, threshold={threshold:.2f}", flush=True)
    
    # Load test S1
    s1_order = []
    s1_by_country = defaultdict(dict)
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4: continue
            eid, name, addr, country = p[0], p[1], p[2], p[3]
            s1_order.append(eid)
            s1_by_country[country][eid] = (name, addr, country)
    
    print(f"  Test S1: {len(s1_order):,}", flush=True)
    for c, recs in sorted(s1_by_country.items()):
        print(f"    {c}: {len(recs):,}", flush=True)
    
    final_matches = {}
    final_candidates = {}
    
    for country in ["France", "US", "India"]:
        s1_country = s1_by_country.get(country, {})
        if not s1_country:
            continue
        
        print(f"\n{'='*70}\n  TEST: {country} ({len(s1_country):,} queries)\n{'='*70}", flush=True)
        
        # Load partners
        partner_raw = {}
        for fn in ["test_source2.tsv", "test_source3.tsv"]:
            partner_raw.update(load_source_file(os.path.join(TEST_DIR, fn), country_filter=country))
        print(f"  {len(partner_raw):,} partners", flush=True)
        
        # Normalize ALL partners (one-time cost per country)
        print(f"  Normalizing partners...", flush=True)
        partner_norm = normalize_all(partner_raw, "partners")
        
        # Fit FeatureBuilder on test partners
        fb = FeatureBuilder()
        fb.fit_idf([str(n.get("name_clean", "")) for n in partner_norm.values()])
        
        # Build inverted index on normalized partners
        # Process S1 in chunks to manage memory
        query_ids = list(s1_country.keys())
        n_chunks = (len(query_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        country_scored = []  # (s1_id, pid, p_cal)
        
        for chunk_idx in range(n_chunks):
            c_start = chunk_idx * CHUNK_SIZE
            c_end = min(c_start + CHUNK_SIZE, len(query_ids))
            chunk_qids = query_ids[c_start:c_end]
            
            print(f"\n  Chunk {chunk_idx+1}/{n_chunks} ({c_start:,}-{c_end:,})...", flush=True)
            
            # Normalize S1 chunk
            s1_chunk_raw = {eid: s1_country[eid] for eid in chunk_qids}
            s1_chunk_norm = normalize_all(s1_chunk_raw, "S1 chunk")
            
            # Blocking
            candidates = build_index_and_retrieve(s1_chunk_norm, partner_norm, country, top_k=TOP_K)
            
            # Store candidates
            for s1_id, pid, _ in candidates:
                if s1_id not in final_candidates:
                    final_candidates[s1_id] = []
                final_candidates[s1_id].append(pid)
            
            # TF-IDF cosine on candidates
            print(f"    Computing TF-IDF cosine for {len(candidates):,} pairs...", flush=True)
            t0 = time.time()
            cosine_scores = compute_tfidf_cosine_batch(candidates, s1_chunk_norm, partner_norm)
            print(f"    Done in {time.time()-t0:.1f}s", flush=True)
            
            # Features
            features = extract_features_batch(candidates, s1_chunk_norm, partner_norm, fb, cosine_scores)
            
            # Inference
            X = features[feature_cols].values
            p_raw = model.predict_proba(X)[:, 1]
            p_cal = cal.transform(p_raw)
            
            for s1_id, pid, pc in zip(features["source1_entity_id"].values,
                                       features["partner_entity_id"].values, p_cal):
                country_scored.append((s1_id, pid, float(pc)))
            
            del features, s1_chunk_norm, s1_chunk_raw, candidates, cosine_scores
            gc.collect()
        
        # Global 1-to-1 for this country
        print(f"\n  Global 1-to-1 ({len(country_scored):,} pairs)...", flush=True)
        scored_df = pd.DataFrame(country_scored, columns=["source1_entity_id", "partner_entity_id", "p_cal"])
        scored_121 = resolve_one_to_one(scored_df, prob_col="p_cal")
        
        passing = scored_121[scored_121["p_cal"] >= threshold]
        country_matches = defaultdict(set)
        for s1, pid in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
            country_matches[s1].add(pid)
        
        # Add empty entries for singletons
        for eid in query_ids:
            if eid not in country_matches:
                country_matches[eid] = set()
        
        n_matched = sum(1 for v in country_matches.values() if v)
        n_links = sum(len(v) for v in country_matches.values())
        print(f"  {country}: {n_matched:,} matched, {n_links:,} links "
              f"({n_matched/len(query_ids):.2%})", flush=True)
        
        final_matches.update(country_matches)
        
        del partner_raw, partner_norm, country_scored, scored_df, scored_121
        gc.collect()
    
    # Write output
    print(f"\n{'='*70}\n  WRITING OUTPUT\n{'='*70}", flush=True)
    
    match_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cand_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    write_tsv_submission(match_file, final_matches, s1_order,
                         "source1_entity_id", "matched_entity_ids")
    write_tsv_submission(cand_file,
                         {eid: set(final_candidates.get(eid, [])) for eid in s1_order},
                         s1_order, "source1_entity_id", "candidate_entity_ids")
    
    n_matched = sum(1 for eid in s1_order if final_matches.get(eid))
    n_links = sum(len(v) for v in final_matches.values())
    print(f"  Rows: {len(s1_order):,}")
    print(f"  Matched: {n_matched:,} ({n_matched/len(s1_order):.2%})")
    print(f"  Singletons: {len(s1_order)-n_matched:,}")
    print(f"  Links: {n_links:,} (avg {n_links/len(s1_order):.2f})")
    
    # Validate
    val_pass, val_msg = run_submission_validator(match_file, cand_file, TEST_DIR)
    
    total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"{'VALIDATION PASSED' if val_pass else 'VALIDATION ISSUES: ' + val_msg}")
    print(f"Total: {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    if args.mode == "train":
        run_train(sample_per_country=args.sample)
    else:
        run_predict()
