"""Benchmark Scipy TF-IDF char n-gram matrix multiplication."""

import os
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

train_dir = os.path.join("dataset", "train")

# Load 5,000 GT
gt_us = {}
s1_ids = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_us[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip())
            s1_ids.append(p[0])
            if len(s1_ids) >= 5000:
                break

needed_s1 = set(s1_ids)
target_partners = {pid for plist in gt_us.values() for pid in plist}

s1_texts = []
s1_list = []
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_list.append(p[0])
            s1_texts.append(p[1].lower() + " " + p[2].lower())

partner_texts = []
partner_list = []
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in target_partners or len(partner_list) < 150000:
                partner_list.append(p[0])
                partner_texts.append(p[1].lower() + " " + p[2].lower())

print(f"Loaded {len(s1_texts):,} queries and {len(partner_texts):,} partner texts.")

t0 = time.time()
print("Vectorizing partner texts with char_wb (3, 4)...")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)
partner_mat = vec.fit_transform(partner_texts)
print(f"Partner matrix shape: {partner_mat.shape} in {time.time() - t0:.2f}s")

t0 = time.time()
q_mat = vec.transform(s1_texts)
print(f"Query matrix shape: {q_mat.shape} in {time.time() - t0:.2f}s")

t0 = time.time()
print("Computing matrix dot product...")
sim_mat = q_mat.dot(partner_mat.T) # (5000, 150000)
print(f"Dot product computed in {time.time() - t0:.2f}s. NNZ: {sim_mat.nnz:,}")

# Fast top-k extraction
t0 = time.time()
indptr = sim_mat.indptr
indices = sim_mat.indices
data = sim_mat.data

hits = 0
total_gt = sum(len(plist) for plist in gt_us.values())
K = 35

for i, s1_id in enumerate(s1_list):
    true_matches = gt_us.get(s1_id, set())
    s, e = indptr[i], indptr[i+1]
    if e > s:
        d = data[s:e]
        cols = indices[s:e]
        if len(d) > K:
            part = np.argpartition(-d, K)[:K]
            top_cols = cols[part]
        else:
            top_cols = cols
        cand_pids = {partner_list[c] for c in top_cols}
        hits += len(true_matches & cand_pids)

print(f"Top-{K} extracted in {time.time() - t0:.2f}s.")
print(f"Recall: {hits / total_gt:.2%} ({hits:,} / {total_gt:,})")
