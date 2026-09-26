"""Phase 1: TF-IDF Char N-Gram Blocking Recall Test at Full Partner Scale.

Tests whether TfidfVectorizer(analyzer='char_wb') achieves >=97% recall
when searching across ALL 6.2M US training partners (not the 150K sample
that gave misleadingly good results).
"""

import os, sys, time
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# Project paths
ROOT = r"c:\Users\sam\Documents\Projects\amazon-ml"
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
src_dir = os.path.join(ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)
from normalize.text import normalize_record

def load_ground_truth(n=10000):
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n:
                break
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            matches = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt[eid] = matches
    return gt

def load_records_raw(filename, country_filter=None):
    """Load records as raw (id, name, addr, country) tuples - NO normalization."""
    records = {}
    with open(os.path.join(TRAIN_DIR, filename), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                if country_filter and p[3] != country_filter:
                    continue
                records[p[0]] = (p[1], p[2], p[3])
    return records

def make_text(name, addr):
    """Create combined text for TF-IDF vectorization (minimal cleaning)."""
    import re, unicodedata
    text = name + " " + addr
    # Strip accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def top_k_per_row_sparse(sim_matrix, k=50):
    """Extract top-K indices and values per row from a sparse matrix."""
    n_rows = sim_matrix.shape[0]
    top_indices = np.zeros((n_rows, k), dtype=np.int32)
    top_values = np.zeros((n_rows, k), dtype=np.float32)
    
    # Convert to CSR for efficient row access
    sim_csr = sim_matrix.tocsr()
    
    for i in range(n_rows):
        row = sim_csr.getrow(i)
        data = row.data
        indices = row.indices
        if len(data) <= k:
            n = len(data)
            top_indices[i, :n] = indices
            top_values[i, :n] = data
        else:
            # Partial sort for top-K
            top_k_idx = np.argpartition(data, -k)[-k:]
            sorted_idx = top_k_idx[np.argsort(-data[top_k_idx])]
            top_indices[i] = indices[sorted_idx]
            top_values[i] = data[sorted_idx]
    
    return top_indices, top_values

print("=" * 70)
print("PHASE 1: TF-IDF Char N-Gram Blocking Recall Test (Full Scale)")
print("=" * 70)

# 1. Load ground truth for first 10K entities
print("\n[1/5] Loading ground truth...", flush=True)
gt = load_ground_truth(10000)

# 2. Load S1 records (US only, first 5K with GT)
print("[2/5] Loading US S1 records...", flush=True)
s1_raw = load_records_raw("train_source1.tsv", country_filter="US")
# Filter to entities that have GT
s1_with_gt = {eid: s1_raw[eid] for eid in gt if eid in s1_raw}
# Take first 5000
s1_sample = dict(list(s1_with_gt.items())[:5000])
print(f"  Selected {len(s1_sample):,} US S1 entities with ground truth", flush=True)

# Compute total true matches
total_true = sum(len(gt[eid]) for eid in s1_sample)
print(f"  Total true match IDs to find: {total_true:,}", flush=True)

# 3. Load ALL US partners
print("[3/5] Loading ALL US partner records...", flush=True)
t0 = time.time()
partner_raw = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    partner_raw.update(load_records_raw(fn, country_filter="US"))
print(f"  Loaded {len(partner_raw):,} US partners in {time.time()-t0:.1f}s", flush=True)

# 4. Vectorize with TF-IDF char n-grams
print("[4/5] TF-IDF vectorization of all partners + queries...", flush=True)
t0 = time.time()

# Prepare partner texts and IDs
partner_ids = list(partner_raw.keys())
partner_texts = [make_text(partner_raw[pid][0], partner_raw[pid][1]) for pid in partner_ids]
n_partners = len(partner_ids)
print(f"  Prepared {n_partners:,} partner texts in {time.time()-t0:.1f}s", flush=True)

# Prepare query texts and IDs
query_ids = list(s1_sample.keys())
query_texts = [make_text(s1_sample[qid][0], s1_sample[qid][1]) for qid in query_ids]
n_queries = len(query_ids)

# Fit vectorizer on partners (they form the search corpus)
print(f"  Fitting TfidfVectorizer on {n_partners:,} texts...", flush=True)
t1 = time.time()
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=80000,
    max_df=0.1,
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32
)
P = vectorizer.fit_transform(partner_texts)
print(f"  Partner matrix: {P.shape}, nnz={P.nnz:,}, density={P.nnz/(P.shape[0]*P.shape[1]):.6f} in {time.time()-t1:.1f}s", flush=True)

# Transform queries
t1 = time.time()
Q = vectorizer.transform(query_texts)
print(f"  Query matrix: {Q.shape}, nnz={Q.nnz:,} in {time.time()-t1:.1f}s", flush=True)

# 5. Chunked sparse matmul + top-K extraction
print("[5/5] Chunked sparse matmul for candidate retrieval...", flush=True)
K = 50  # top-K candidates per query
PARTNER_CHUNK = 300000  # process 300K partners at a time

# Build partner ID to index mapping
pid_to_idx = {pid: i for i, pid in enumerate(partner_ids)}

all_top_indices = np.zeros((n_queries, K), dtype=np.int32)
all_top_values = np.zeros((n_queries, K), dtype=np.float32)

n_p_chunks = (n_partners + PARTNER_CHUNK - 1) // PARTNER_CHUNK
t_start = time.time()

for pc in range(n_p_chunks):
    p_start = pc * PARTNER_CHUNK
    p_end = min(p_start + PARTNER_CHUNK, n_partners)
    P_chunk = P[p_start:p_end]
    
    t_chunk = time.time()
    sim = Q.dot(P_chunk.T)  # n_queries x chunk_size sparse
    
    # Extract top-K per row from this chunk
    chunk_top_idx, chunk_top_val = top_k_per_row_sparse(sim, K)
    
    # Adjust indices to global partner indices
    chunk_top_idx += p_start
    
    # Merge with existing top-K
    for i in range(n_queries):
        # Combine current top-K with chunk top-K
        combined_idx = np.concatenate([all_top_indices[i], chunk_top_idx[i]])
        combined_val = np.concatenate([all_top_values[i], chunk_top_val[i]])
        
        # Keep top-K
        if np.count_nonzero(combined_val) > K:
            top_k = np.argpartition(combined_val, -K)[-K:]
            sorted_top = top_k[np.argsort(-combined_val[top_k])]
            all_top_indices[i] = combined_idx[sorted_top]
            all_top_values[i] = combined_val[sorted_top]
        else:
            # Just keep what we have, sorted
            mask = combined_val > 0
            n_valid = mask.sum()
            valid_idx = np.argsort(-combined_val)[:n_valid]
            all_top_indices[i, :n_valid] = combined_idx[valid_idx]
            all_top_values[i, :n_valid] = combined_val[valid_idx]
    
    elapsed = time.time() - t_chunk
    print(f"  Partner chunk {pc+1}/{n_p_chunks} ({p_start:,}-{p_end:,}) processed in {elapsed:.1f}s", flush=True)

total_time = time.time() - t_start
print(f"  Total retrieval time: {total_time:.1f}s", flush=True)

# 6. Evaluate recall
print("\n" + "=" * 70)
print("RECALL EVALUATION")
print("=" * 70)

found = 0
missed = 0
missed_examples = []

for i, qid in enumerate(query_ids):
    true_matches = gt.get(qid, set())
    if not true_matches:
        continue
    
    # Get candidate IDs
    cand_pids = set()
    for j in range(K):
        if all_top_values[i, j] > 0:
            cand_pids.add(partner_ids[all_top_indices[i, j]])
    
    for true_pid in true_matches:
        if true_pid in cand_pids:
            found += 1
        else:
            missed += 1
            if len(missed_examples) < 10:
                # Check if the true partner even exists in our pool
                in_pool = true_pid in pid_to_idx
                missed_examples.append((qid, true_pid, in_pool))

recall = found / (found + missed) if (found + missed) > 0 else 0
print(f"\nCandidate Recall @ Top-{K}: {recall:.4f} ({recall:.2%})")
print(f"  Found: {found:,} / {found + missed:,}")
print(f"  Missed: {missed:,}")
print(f"\nAvg candidates per query: {np.count_nonzero(all_top_values, axis=1).mean():.1f}")

if missed_examples:
    print(f"\nSample missed matches (first 10):")
    for qid, pid, in_pool in missed_examples:
        q_text = make_text(s1_sample[qid][0], s1_sample[qid][1])[:60]
        if in_pool:
            p_text = make_text(partner_raw[pid][0], partner_raw[pid][1])[:60]
        else:
            p_text = "[NOT IN POOL]"
        print(f"  Q: {qid} -> '{q_text}'")
        print(f"  P: {pid} -> '{p_text}' (in_pool={in_pool})")
        print()

# Test at different K values
for test_k in [10, 25, 50]:
    test_found = 0
    test_total = 0
    for i, qid in enumerate(query_ids):
        true_matches = gt.get(qid, set())
        if not true_matches:
            continue
        cand_pids = set()
        for j in range(test_k):
            if all_top_values[i, j] > 0:
                cand_pids.add(partner_ids[all_top_indices[i, j]])
        for true_pid in true_matches:
            test_total += 1
            if true_pid in cand_pids:
                test_found += 1
    test_recall = test_found / test_total if test_total > 0 else 0
    print(f"  Recall @ Top-{test_k:>2}: {test_recall:.4f} ({test_recall:.2%})")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("=" * 70)
