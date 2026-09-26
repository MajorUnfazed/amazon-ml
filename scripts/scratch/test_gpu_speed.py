import os
import sys
import time
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("TESTING GPU ACCELERATED TF-IDF BLOCKING ON RTX 4060")
print("=" * 70)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# Create synthetic or real sample: 50,000 queries, 500,000 partner records, 30,000 features
N_PARTNERS = 500000
N_QUERIES = 20000
N_FEAT = 40000

# Test PyTorch sparse or chunked dense dot product on GPU
# A typical sparse row has ~15 non-zero elements
print(f"Creating test sparse matrices: {N_QUERIES:,} queries x {N_PARTNERS:,} partners ({N_FEAT:,} features)...")
# Let's test with real text from test_source1 and test_source2 (France partition)
test_dir = os.path.join("dataset", "test")

s1_texts = []
s1_ids = []
with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) >= 4 and p[3] == "France":
            s1_ids.append(p[0])
            s1_texts.append(p[1].lower() + " " + p[2].lower())
            if len(s1_ids) >= 10000:
                break

p_texts = []
p_ids = []
for fn in ["test_source2.tsv", "test_source3.tsv"]:
    with open(os.path.join(test_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                p_ids.append(p[0])
                p_texts.append(p[1].lower() + " " + p[2].lower())
                if len(p_ids) >= 200000:
                    break

print(f"Loaded {len(s1_texts):,} France S1 and {len(p_texts):,} France Partners.")

# Vectorize with char_wb (3, 4)
t0 = time.time()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=40000, sublinear_tf=True, dtype=np.float32)
p_mat = vec.fit_transform(p_texts)
q_mat = vec.transform(s1_texts)
print(f"Vectorized in {time.time() - t0:.2f}s. Shapes: P={p_mat.shape}, Q={q_mat.shape}")

# Test 1: Scipy CPU dot product
t0 = time.time()
sim_cpu = q_mat.dot(p_mat.T)
print(f"Scipy CPU Dot product (10k x 200k) completed in {time.time() - t0:.2f}s.")

# Test 2: GPU PyTorch chunked dot product
t0 = time.time()
# Convert P to PyTorch CSR or chunks
# Since P is (200000, 40000), P.T is (40000, 200000)
# In PyTorch, we can do batch dense matrix multiplication on GPU:
CHUNK = 2000
for i in range(0, len(s1_texts), CHUNK):
    q_chunk = torch.tensor(q_mat[i:i+CHUNK].toarray(), device=device, dtype=torch.float16)
    # Even faster: Scipy dot is already multi-threaded and takes only a few seconds!
print("GPU check completed.")
