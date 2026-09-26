import os
import sys
import time
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("TESTING PYTORCH GPU SPARSE/HALF MATMUL ON RTX 4060")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, torch.cuda.get_device_name(0))

# Create 5,000 queries x 150,000 partners, 40,000 features
# Test GPU dense-chunk matmul:
# P.T is (40000, 150000). In float16, that is 40,000 * 150,000 * 2 bytes = 12 GB.
# But in chunks of 50,000 partners: 40,000 * 50,000 * 2 = 4 GB! Fits easily in 8.5 GB VRAM!
# Or test PyTorch sparse tensor:
# Q is (5000, 40000) sparse, P is (150000, 40000) sparse.
# Let's benchmark Scipy CSR dot vs PyTorch GPU half matmul!

# Generate realistic text
s1_texts = ["company name llc 123 main street springfield il" for _ in range(5000)]
p_texts = ["company name corp 123 main st springfield illinois" for _ in range(150000)]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=30000, sublinear_tf=True, dtype=np.float32)
p_mat = vec.fit_transform(p_texts)
q_mat = vec.transform(s1_texts)

print("P shape:", p_mat.shape, "NNZ:", p_mat.nnz)
print("Q shape:", q_mat.shape, "NNZ:", q_mat.nnz)

# Test 1: Scipy CPU dot
t0 = time.time()
sim_cpu = q_mat.dot(p_mat.T)
print(f"Scipy CPU Dot product took {time.time() - t0:.2f}s.")

# Test 2: PyTorch GPU Chunked Matmul
t0 = time.time()
# Convert Q chunk to GPU float16
q_gpu = torch.from_numpy(q_mat[:1000].toarray()).to(device=device, dtype=torch.float16)

# Test partner chunk on GPU (30,000 partners)
p_chunk_gpu = torch.from_numpy(p_mat[:30000].toarray().T).to(device=device, dtype=torch.float16)

torch.cuda.synchronize()
t_gpu_start = time.time()
res = torch.matmul(q_gpu, p_chunk_gpu)
torch.cuda.synchronize()
t_gpu_end = time.time()
print(f"PyTorch GPU matmul (1,000 queries x 30,000 partners) took: {t_gpu_end - t_gpu_start:.4f}s!")
print(f"Projected GPU throughput: {1000 * 30000 / (t_gpu_end - t_gpu_start):,.0f} pair-scores/sec")
print("=" * 70)
