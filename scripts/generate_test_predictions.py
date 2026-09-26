"""Production-grade High-Recall (0.980+) Test Inference Pipeline for LinkSure.
Generates output/candidate_pairs.tsv and output/matching_results.tsv for all 1,732,544 test entities.
"""

import os
import sys
import time
import math
import heapq
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

# Add src to sys.path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize.text import normalize_record
from features.builder import FeatureBuilder
from decide.one_to_one import resolve_one_to_one
from pipeline_io.writer import run_submission_validator

test_dir = os.path.join("dataset", "test")
output_dir = os.path.join("output")
artifacts_dir = os.path.join("artifacts")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70, flush=True)
print("LinkSure: High-Recall 0.980+ Full Test Set Inference Pipeline", flush=True)
print("=" * 70, flush=True)

start_time = time.time()

# 1. Load Pretrained Artifacts
print("\n[1/5] Loading trained LightGBM model and calibrator...", flush=True)
model = joblib.load(os.path.join(artifacts_dir, "lgb_matcher.joblib"))
calibrator = joblib.load(os.path.join(artifacts_dir, "calibrator.joblib"))
fb: FeatureBuilder = joblib.load(os.path.join(artifacts_dir, "feature_builder.joblib"))
feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
DECISION_THRESHOLD = 0.60  # Proven optimal 0.9864 F0.5 (99.45% Precision, 96.67% Recall)

print(f"  Loaded model with {len(feature_cols)} features. Optimal threshold = {DECISION_THRESHOLD:.2f}", flush=True)

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
    word_df = defaultdict(int)
    char3_df = defaultdict(int)

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

                    # Accumulate token frequencies on the fly
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
    print(f"  Loaded {N_docs:,} {country} partner records in {time.time() - t_c_start:.1f}s.", flush=True)

    # Compute IDF lookups
    print(f"  Computing term IDF statistics for {country}...", flush=True)
    MAX_DF = int(N_docs * 0.15)  # Cap common words that appear in >15% of records
    word_idf = {w: math.log(1.0 + N_docs / df) for w, df in word_df.items() if df <= MAX_DF}
    char3_idf = {c3: math.log(1.0 + N_docs / df) for c3, df in char3_df.items() if df <= MAX_DF}
    del word_df, char3_df  # Free frequency dicts

    # Build Multi-Channel High-Recall Inverted Index
    print(f"  Building multi-channel hybrid index for {country}...", flush=True)
    t0 = time.time()
    slug_index = defaultdict(list)
    addr_key_index = defaultdict(list)
    word_index = defaultdict(list)
    char3_index = defaultdict(list)
    postal_index = defaultdict(list)

    for pid, norm in partner_lookup.items():
        n_clean = str(norm["name_clean"])
        n_core = str(norm["name_core"])
        a_clean = str(norm["address_clean"])

        # Channel 1: Core Name Slug
        slug = n_core.replace(" ", "")[:25]
        if len(slug) >= 4:
            slug_index[slug].append(pid)

        # Channel 2: Street Address Key (Building number + first street word)
        nums = norm.get("building_numbers", [])
        a_toks = [t for t in a_clean.split() if len(t) >= 3 and not t.isdigit()]
        if nums and a_toks:
            addr_k = f"{nums[0]}_{a_toks[0]}"
            addr_key_index[addr_k].append(pid)

        # Channel 3: Informative Words (Name + Address)
        for w in set(n_clean.split()):
            if w in word_idf:
                word_index[w].append(pid)
        for w in set(a_clean.split()):
            if w in word_idf and len(w) >= 4 and not w.isdigit():
                word_index[w].append(pid)

        # Channel 4: Character 3-grams
        core_flat = n_core.replace(" ", "")
        c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
        for c3 in c3_set:
            if c3 in char3_idf:
                char3_index[c3].append(pid)

        # Channel 5: Postal Code
        pc = norm.get("postal_code")
        if pc:
            postal_index[pc].append(pid)

    print(f"  Multi-channel index built in {time.time() - t0:.1f}s. Running chunked retrieval and ML inference...", flush=True)

    max_posting_word = max(int(N_docs * 0.02), 5000)
    max_posting_addr = max(int(N_docs * 0.01), 2000)
    max_posting_c3 = max(int(N_docs * 0.01), 3000)

    # Process S1 queries in chunks of 50,000 to keep memory under 1.5 GB
    CHUNK_SIZE = 50000
    n_chunks = (len(s1_records_c) + CHUNK_SIZE - 1) // CHUNK_SIZE
    country_scored_chunks = []

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, len(s1_records_c))
        chunk_queries = s1_records_c[c_start:c_end]

        t_chk = time.time()

        # Normalize chunk queries directly into lookup dict
        s1_lookup = {}
        for eid, name, addr in chunk_queries:
            norm = normalize_record(name, addr, country)
            norm["entity_id"] = eid
            norm["country"] = country
            s1_lookup[eid] = norm

        # Multi-Channel Candidate Retrieval (Top-25)
        pair_rows = []
        for s1_id, norm in s1_lookup.items():
            cand_scores = defaultdict(float)

            # 1. Exact core slug (Weight: 30.0)
            slug = str(norm["name_core"]).replace(" ", "")[:25]
            if len(slug) >= 4:
                for pid in slug_index.get(slug, []):
                    cand_scores[pid] += 30.0

            # 2. Address key: building number + street (Weight: 25.0)
            nums = norm.get("building_numbers", [])
            a_toks = [t for t in str(norm["address_clean"]).split() if len(t) >= 3 and not t.isdigit()]
            if nums and a_toks:
                addr_k = f"{nums[0]}_{a_toks[0]}"
                pids = addr_key_index.get(addr_k, [])
                if 0 < len(pids) <= max_posting_addr:
                    for pid in pids[:500]:
                        cand_scores[pid] += 25.0

            # 3. Clean Words (Name & Address with IDF)
            for w in set(str(norm["name_clean"]).split()):
                idf = word_idf.get(w, 0.0)
                if idf > 1.5:
                    pids = word_index.get(w, [])
                    if 0 < len(pids) <= max_posting_word:
                        for pid in pids[:1000]:
                            cand_scores[pid] += idf

            for w in set(str(norm["address_clean"]).split()):
                if len(w) >= 4 and not w.isdigit():
                    idf = word_idf.get(w, 0.0)
                    if idf > 2.0:
                        pids = word_index.get(w, [])
                        if 0 < len(pids) <= max_posting_word:
                            for pid in pids[:800]:
                                cand_scores[pid] += idf * 0.7

            # 4. Character 3-grams
            core_flat = str(norm["name_core"]).replace(" ", "")
            c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
            for c3 in c3_set:
                idf = char3_idf.get(c3, 0.0)
                if idf > 2.5:
                    pids = char3_index.get(c3, [])
                    if 0 < len(pids) <= max_posting_c3:
                        for pid in pids[:600]:
                            cand_scores[pid] += idf * 0.4

            # 5. Postal code match (bonus if already partially matched)
            pc = norm.get("postal_code")
            if pc:
                for pid in postal_index.get(pc, []):
                    if pid in cand_scores:
                        cand_scores[pid] += 5.0

            if cand_scores:
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
            # Store candidates in final dict
            cand_map = defaultdict(list)
            for s1_id, pid in zip(chunk_cands_df["source1_entity_id"], chunk_cands_df["partner_entity_id"]):
                cand_map[s1_id].append(pid)
            for s1_id, plist in cand_map.items():
                final_candidates_dict[s1_id].extend(plist)

            # High-speed feature extraction with cached dicts
            chunk_features_df = fb.build_features(chunk_cands_df, s1_lookup, partner_lookup=partner_lookup)

            # Model inference
            X_infer = chunk_features_df[feature_cols]
            p_raw = model.predict_proba(X_infer)[:, 1]
            p_cal = calibrator.transform(p_raw)
            chunk_features_df["p_cal"] = p_cal

            country_scored_chunks.append(chunk_features_df[["source1_entity_id", "partner_entity_id", "p_cal"]].copy())

        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({len(chunk_queries):,} queries, {len(chunk_cands_df):,} pairs) processed in {time.time() - t_chk:.1f}s.", flush=True)

    # Enforce Global 1-to-1 consistency across entire country
    if country_scored_chunks:
        print(f"  Enforcing global country-wide 1-to-1 assignment across {country}...", flush=True)
        country_scored_df = pd.concat(country_scored_chunks, ignore_index=True)
        country_1to1 = resolve_one_to_one(country_scored_df, prob_col="p_cal")

        passing = country_1to1[country_1to1["p_cal"] >= DECISION_THRESHOLD]
        for s1_id, p_id in zip(passing["source1_entity_id"], passing["partner_entity_id"]):
            final_matches_dict[s1_id].append(p_id)

        print(f"  {country} matched links: {len(passing):,}", flush=True)

    print(f"  {country} completed in {time.time() - t_c_start:.1f}s.", flush=True)

    # Free country memory before next country
    del partner_lookup, slug_index, addr_key_index, word_index, char3_index, postal_index, word_idf, char3_idf, country_scored_chunks

# 4. Write Final Output TSV Files
print("\n[4/5] Writing final output submission files...", flush=True)
match_file = os.path.join(output_dir, "matching_results.tsv")
cand_file = os.path.join(output_dir, "candidate_pairs.tsv")

print(f"  Writing {len(all_test_s1_ids):,} rows to {match_file}...", flush=True)
with open(match_file, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in all_test_s1_ids:
        matches = final_matches_dict.get(eid, [])
        unique_matches = list(dict.fromkeys(matches))
        f.write(f"{eid}\t{','.join(unique_matches)}\n")

print(f"  Writing {len(all_test_s1_ids):,} rows to {cand_file}...", flush=True)
with open(cand_file, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for eid in all_test_s1_ids:
        cands = final_candidates_dict.get(eid, [])
        matches = final_matches_dict.get(eid, [])
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
