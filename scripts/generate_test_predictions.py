"""Chunked, memory-safe test inference pipeline for LinkSure.
Generates output/candidate_pairs.tsv and output/matching_results.tsv for all 1,732,544 test entities.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import heapq
from collections import defaultdict

# Add src to sys.path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize.text import normalize_record
from features.builder import FeatureBuilder
from decide.one_to_one import resolve_one_to_one
from decide.expected_f import decide_global_threshold
from pipeline_io.writer import write_tsv_submission, run_submission_validator

test_dir = os.path.join("dataset", "test")
output_dir = os.path.join("output")
artifacts_dir = os.path.join("artifacts")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70, flush=True)
print("LinkSure: Full Test Set Inference Pipeline", flush=True)
print("=" * 70, flush=True)

start_time = time.time()

# 1. Load Pretrained Artifacts
print("\n[1/5] Loading trained LightGBM model and calibrator...", flush=True)
model = joblib.load(os.path.join(artifacts_dir, "lgb_matcher.joblib"))
calibrator = joblib.load(os.path.join(artifacts_dir, "calibrator.joblib"))
fb: FeatureBuilder = joblib.load(os.path.join(artifacts_dir, "feature_builder.joblib"))
feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
try:
    threshold = joblib.load(os.path.join(artifacts_dir, "best_threshold.joblib"))
except Exception:
    threshold = 0.50

print(f"  Loaded model with {len(feature_cols)} features. Decision threshold = {threshold:.2f}", flush=True)

# 2. Read Test Source 1 IDs to ensure 100% coverage
print("\n[2/5] Reading all test Source 1 entities...", flush=True)
all_test_s1_ids = []
s1_by_country = defaultdict(list)

with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        eid, name, addr, country = p[0], p[1], p[2], p[3]
        all_test_s1_ids.append(eid)
        s1_by_country[country].append((eid, name, addr))

total_s1 = len(all_test_s1_ids)
print(f"  Total Test S1 Entities: {total_s1:,}", flush=True)
for c, recs in s1_by_country.items():
    print(f"    - {c:<10}: {len(recs):,} entities ({len(recs)/total_s1:.2%})", flush=True)

final_candidates_dict = {eid: [] for eid in all_test_s1_ids}
final_matches_dict = {eid: [] for eid in all_test_s1_ids}

# 3. Process Each Country Partition Separately
# Order: France first (smallest), then US, then India
countries_to_process = ["France", "US", "India"]

for country in countries_to_process:
    s1_records_c = s1_by_country.get(country, [])
    if not s1_records_c:
        continue

    print(f"\n[3/5] Processing Partition: {country.upper()} ({len(s1_records_c):,} S1 queries)...", flush=True)
    t_c_start = time.time()

    # Load partner records for this country
    print(f"  Loading {country} partner pool from test_source2 and test_source3...", flush=True)
    partner_lookup = {}

    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        with open(os.path.join(test_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4 and p[3] == country:
                    norm = normalize_record(p[1], p[2], p[3])
                    norm["entity_id"] = p[0]
                    norm["country"] = country
                    partner_lookup[p[0]] = norm

    print(f"  Loaded {len(partner_lookup):,} {country} partner records in {time.time() - t_c_start:.1f}s.", flush=True)

    # Build Inverted Indexes for Country
    print(f"  Building inverted index for {country}...", flush=True)
    t0 = time.time()
    token_index = defaultdict(list)
    slug_index = defaultdict(list)

    for pid, norm in partner_lookup.items():
        n_clean = norm["name_clean"]
        n_core = norm["name_core"]
        a_clean = norm["address_clean"]
        toks = set(str(n_clean).split() + str(a_clean).split()[:3])
        for t in toks:
            if len(t) >= 2:
                token_index[t].append(pid)
        slug = str(n_core).replace(" ", "")[:25]
        if len(slug) >= 4:
            slug_index[slug].append(pid)

    print(f"  Index built in {time.time() - t0:.1f}s. Running chunked retrieval and scoring...", flush=True)

    # Process S1 queries in chunks of 50,000 to keep memory under 1 GB
    CHUNK_SIZE = 50000
    n_chunks = (len(s1_records_c) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, len(s1_records_c))
        chunk_queries = s1_records_c[c_start:c_end]

        t_chk = time.time()

        # Normalize chunk queries directly into fast lookup dict
        s1_lookup = {}
        for eid, name, addr in chunk_queries:
            norm = normalize_record(name, addr, country)
            norm["entity_id"] = eid
            norm["country"] = country
            s1_lookup[eid] = norm

        # Candidate Retrieval
        pair_rows = []
        for s1_id, norm in s1_lookup.items():
            n_clean = norm["name_clean"]
            n_core = norm["name_core"]
            a_clean = norm["address_clean"]
            cand_scores = defaultdict(float)

            slug = str(n_core).replace(" ", "")[:25]
            if len(slug) >= 4:
                for pid in slug_index.get(slug, []):
                    cand_scores[pid] += 8.0

            toks = set(str(n_clean).split() + str(a_clean).split()[:3])
            for t in toks:
                pids = token_index.get(t, [])
                if 0 < len(pids) <= 300:
                    w = 3.0 if t in str(n_clean) else 1.0
                    for pid in pids:
                        cand_scores[pid] += w

            if cand_scores:
                if len(cand_scores) <= 25:
                    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
                else:
                    top_cands = heapq.nlargest(25, cand_scores.items(), key=lambda x: x[1])

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

        chunk_cands_df = pd.DataFrame(pair_rows)

        if len(chunk_cands_df) > 0:
            # Store candidates in final dict using fast dict loop
            cand_map = defaultdict(list)
            for s1_id, pid in zip(chunk_cands_df["source1_entity_id"], chunk_cands_df["partner_entity_id"]):
                cand_map[s1_id].append(pid)
            for s1_id, plist in cand_map.items():
                final_candidates_dict[s1_id] = plist

            # High-speed feature extraction with cached dicts
            chunk_features_df = fb.build_features(chunk_cands_df, s1_lookup, partner_lookup=partner_lookup)

            # Model inference
            X_infer = chunk_features_df[feature_cols]
            p_raw = model.predict_proba(X_infer)[:, 1]
            p_cal = calibrator.transform(p_raw)

            chunk_features_df["p_cal"] = p_cal

            # Enforce 1-to-1 consistency within chunk
            chunk_1to1 = resolve_one_to_one(chunk_features_df, prob_col="p_cal")

            # Apply precision-heavy threshold
            passing = chunk_1to1[chunk_1to1["p_cal"] >= threshold]
            for s1_id, p_id in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
                final_matches_dict[s1_id].append(p_id)

        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({len(chunk_queries):,} queries, {len(chunk_cands_df):,} pairs) processed in {time.time() - t_chk:.1f}s.", flush=True)

    print(f"  {country} completed in {time.time() - t_c_start:.1f}s.", flush=True)

# 4. Write Final Output TSV Files
print("\n[4/5] Writing final output submission files...", flush=True)
match_file = os.path.join(output_dir, "matching_results.tsv")
cand_file = os.path.join(output_dir, "candidate_pairs.tsv")

print(f"  Writing {len(all_test_s1_ids):,} rows to {match_file}...", flush=True)
with open(match_file, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in all_test_s1_ids:
        matches = final_matches_dict.get(eid, [])
        # Deduplicate while preserving order
        unique_matches = list(dict.fromkeys(matches))
        f.write(f"{eid}\t{','.join(unique_matches)}\n")

print(f"  Writing {len(all_test_s1_ids):,} rows to {cand_file}...", flush=True)
with open(cand_file, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for eid in all_test_s1_ids:
        cands = final_candidates_dict.get(eid, [])
        matches = final_matches_dict.get(eid, [])
        # Ensure matches are always a subset of candidates
        all_cands = list(dict.fromkeys(cands + matches))
        f.write(f"{eid}\t{','.join(all_cands)}\n")

print(f"  Both TSV files written successfully.", flush=True)

# 5. Run Official Validator
print("\n[5/5] Running official challenge validator...", flush=True)
val_pass, val_msg = run_submission_validator(
    matching_tsv_path=match_file,
    candidate_tsv_path=cand_file,
    test_dir=test_dir
)

total_elapsed = time.time() - start_time
print("=" * 70, flush=True)
if val_pass:
    print(f"SUBMISSION VALIDATION: SUCCESS (PASS)", flush=True)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} mins).", flush=True)
else:
    print(f"SUBMISSION VALIDATION ISSUES:\n{val_msg}", flush=True)
print("=" * 70, flush=True)
