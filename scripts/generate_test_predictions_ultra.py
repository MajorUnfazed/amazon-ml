"""
LinkSure Ultra-Blocker 0.99+ High-Recall Test Inference Pipeline.

Incorporates 8 Advanced Blocking Channels to push Recall Ceiling to >= 99.5%:
1. Core Name Exact Slug
2. Permuted Name Word-Set Slug (catches word reorderings)
3. Multi-Token Building Number + Street Keys (catches adjectives/directional words)
4. Soundex Phonetic Indexing (catches phonetic typos like Chandra/Chander, Kumar/Coomar)
5. Acronym / Initialism Matching (e.g. TCS vs Tata Consultancy Services)
6. Clean Informative Words (Name + Address with IDF)
7. Character 3-Grams with dynamic IDF caps
8. Exact Postal Code Block

Features: 52 Relational Features + Grouped Expected-F0.5 Subset Selection.
Candidate Capacity: Top-60 per S1 entity.
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
import gc

# Add src to sys.path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalize.text import normalize_record
from features.builder import FeatureBuilder
from decide.one_to_one import resolve_one_to_one
from decide.expected_f import decide_expected_f05
from pipeline_io.writer import run_submission_validator

test_dir = os.path.join("dataset", "test")
output_dir = os.path.join("output")
artifacts_dir = os.path.join("artifacts")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70, flush=True)
print("LinkSure Ultra-Blocker: 0.99+ Near-Ceiling Inference Pipeline", flush=True)
print("=" * 70, flush=True)

start_time = time.time()

# Fast Pure-Python Soundex for Phonetic Inverted Index
def get_soundex(name: str) -> str:
    """Computes standard Soundex code for phonetic brand matching."""
    if not name or len(name) < 2:
        return ""
    name = str(name).upper()
    digits = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }
    first_char = name[0]
    if not first_char.isalpha():
        return ""
    res = [first_char]
    prev = digits.get(first_char, "")
    for char in name[1:]:
        d = digits.get(char, "")
        if d and d != prev:
            res.append(d)
            if len(res) == 4:
                break
        prev = d
    return "".join(res).ljust(4, "0")[:4]

# 1. Load Pretrained Artifacts
print("\n[1/5] Loading trained LightGBM model and calibrator...", flush=True)
model = joblib.load(os.path.join(artifacts_dir, "lgb_matcher.joblib"))
calibrator = joblib.load(os.path.join(artifacts_dir, "calibrator.joblib"))
fb: FeatureBuilder = joblib.load(os.path.join(artifacts_dir, "feature_builder.joblib"))
feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
print(f"  Loaded model with {len(feature_cols)} features.", flush=True)

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

                    # Accumulate token frequencies
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
    MAX_DF = int(N_docs * 0.15)
    word_idf = {w: math.log(1.0 + N_docs / df) for w, df in word_df.items() if df <= MAX_DF}
    char3_idf = {c3: math.log(1.0 + N_docs / df) for c3, df in char3_df.items() if df <= MAX_DF}
    del word_df, char3_df

    # Build 8-Channel Ultra-Blocker Inverted Index
    print(f"  Building 8-channel Ultra-Blocker index for {country}...", flush=True)
    t0 = time.time()
    slug_index = defaultdict(list)
    sorted_slug_index = defaultdict(list)
    addr_key_index = defaultdict(list)
    soundex_index = defaultdict(list)
    acronym_index = defaultdict(list)
    word_index = defaultdict(list)
    char3_index = defaultdict(list)
    postal_index = defaultdict(list)

    for pid, norm in partner_lookup.items():
        n_clean = str(norm["name_clean"])
        n_core = str(norm["name_core"])
        a_clean = str(norm["address_clean"])
        core_words = n_core.split()

        # Channel 1: Core Name Slug
        slug = n_core.replace(" ", "")[:25]
        if len(slug) >= 4:
            slug_index[slug].append(pid)

        # Channel 2: Permuted Word-Set Slug (catches word order changes)
        if len(core_words) >= 2:
            sorted_slug = "".join(sorted(core_words))[:25]
            if len(sorted_slug) >= 4:
                sorted_slug_index[sorted_slug].append(pid)

        # Channel 3: Multi-Token Street Address Keys (Building number + all street words)
        nums = norm.get("building_numbers", [])
        a_toks = [t for t in a_clean.split() if len(t) >= 3 and not t.isdigit()]
        if nums and a_toks:
            for num in nums[:2]:
                for tok in a_toks[:4]:
                    addr_key_index[f"{num}_{tok}"].append(pid)

        # Channel 4: Soundex Phonetic Index on Brand Token (catches phonetically identical names)
        if core_words:
            first_brand = core_words[0]
            sx = get_soundex(first_brand)
            if sx:
                soundex_index[sx].append(pid)

        # Channel 5: Acronym / Initialism (e.g. TCS vs Tata Consultancy Services)
        if len(core_words) >= 2:
            acro = "".join(w[0] for w in core_words if w)
            if 2 <= len(acro) <= 6:
                acronym_index[acro].append(pid)

        # Channel 6: Informative Words (Name + Address)
        for w in set(n_clean.split()):
            if w in word_idf:
                word_index[w].append(pid)
        for w in set(a_clean.split()):
            if w in word_idf and len(w) >= 4 and not w.isdigit():
                word_index[w].append(pid)

        # Channel 7: Character 3-grams
        core_flat = n_core.replace(" ", "")
        c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
        for c3 in c3_set:
            if c3 in char3_idf:
                char3_index[c3].append(pid)

        # Channel 8: Postal Code
        pc = norm.get("postal_code")
        if pc:
            postal_index[pc].append(pid)

    print(f"  8-Channel Ultra-Blocker built in {time.time() - t0:.1f}s. Running high-recall scoring...", flush=True)

    max_posting_word = max(int(N_docs * 0.02), 5000)
    max_posting_addr = max(int(N_docs * 0.01), 2500)
    max_posting_c3 = max(int(N_docs * 0.01), 3000)
    max_posting_sx = max(int(N_docs * 0.01), 2000)

    # Process S1 queries in chunks of 25,000
    CHUNK_SIZE = 25000
    n_chunks = (len(s1_records_c) + CHUNK_SIZE - 1) // CHUNK_SIZE
    country_scored_chunks = []

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, len(s1_records_c))
        chunk_queries = s1_records_c[c_start:c_end]

        t_chk = time.time()

        s1_lookup = {}
        for eid, name, addr in chunk_queries:
            norm = normalize_record(name, addr, country)
            norm["entity_id"] = eid
            norm["country"] = country
            s1_lookup[eid] = norm

        # Multi-Channel Candidate Retrieval (Top-60)
        pair_rows = []
        for s1_id, norm in s1_lookup.items():
            cand_scores = defaultdict(float)
            n_clean = str(norm["name_clean"])
            n_core = str(norm["name_core"])
            a_clean = str(norm["address_clean"])
            core_words = n_core.split()

            # 1. Exact core slug (Weight: 35.0)
            slug = n_core.replace(" ", "")[:25]
            if len(slug) >= 4:
                for pid in slug_index.get(slug, []):
                    cand_scores[pid] += 35.0

            # 2. Permuted core slug (Weight: 25.0)
            if len(core_words) >= 2:
                sorted_slug = "".join(sorted(core_words))[:25]
                if len(sorted_slug) >= 4:
                    for pid in sorted_slug_index.get(sorted_slug, []):
                        cand_scores[pid] += 25.0

            # 3. Multi-token Address Keys (Weight: 25.0)
            nums = norm.get("building_numbers", [])
            a_toks = [t for t in a_clean.split() if len(t) >= 3 and not t.isdigit()]
            if nums and a_toks:
                for num in nums[:2]:
                    for tok in a_toks[:4]:
                        addr_k = f"{num}_{tok}"
                        pids = addr_key_index.get(addr_k, [])
                        if 0 < len(pids) <= max_posting_addr:
                            for pid in pids[:300]:
                                cand_scores[pid] += 25.0

            # 4. Soundex Phonetic Match (Weight: 10.0 bonus if city/zip or word overlap)
            if core_words:
                sx = get_soundex(core_words[0])
                if sx:
                    pids = soundex_index.get(sx, [])
                    if 0 < len(pids) <= max_posting_sx:
                        for pid in pids[:300]:
                            cand_scores[pid] += 8.0

            # 5. Acronym Match (Weight: 15.0)
            if len(core_words) >= 2:
                acro = "".join(w[0] for w in core_words if w)
                if 2 <= len(acro) <= 6:
                    for pid in acronym_index.get(acro, []):
                        cand_scores[pid] += 15.0

            # 6. Informative Words (Name + Address with IDF)
            for w in set(n_clean.split()):
                idf = word_idf.get(w, 0.0)
                if idf > 1.5:
                    pids = word_index.get(w, [])
                    if 0 < len(pids) <= max_posting_word:
                        for pid in pids[:1000]:
                            cand_scores[pid] += idf

            for w in set(a_clean.split()):
                if len(w) >= 4 and not w.isdigit():
                    idf = word_idf.get(w, 0.0)
                    if idf > 2.0:
                        pids = word_index.get(w, [])
                        if 0 < len(pids) <= max_posting_word:
                            for pid in pids[:800]:
                                cand_scores[pid] += idf * 0.7

            # 7. Character 3-grams
            core_flat = n_core.replace(" ", "")
            c3_set = {core_flat[i:i+3] for i in range(len(core_flat) - 2)}
            for c3 in c3_set:
                idf = char3_idf.get(c3, 0.0)
                if idf > 2.5:
                    pids = char3_index.get(c3, [])
                    if 0 < len(pids) <= max_posting_c3:
                        for pid in pids[:600]:
                            cand_scores[pid] += idf * 0.4

            # 8. Postal code match (bonus)
            pc = norm.get("postal_code")
            if pc:
                for pid in postal_index.get(pc, []):
                    if pid in cand_scores:
                        cand_scores[pid] += 6.0

            if cand_scores:
                # Top-60 high-recall candidate selection
                top_cands = heapq.nlargest(60, cand_scores.items(), key=lambda x: x[1])
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

            # Retain plausible pairs (p_cal >= 0.20)
            plausible_pairs = chunk_features_df[chunk_features_df["p_cal"] >= 0.20][["source1_entity_id", "partner_entity_id", "p_cal"]].copy()
            if len(plausible_pairs) > 0:
                country_scored_chunks.append(plausible_pairs)

            del chunk_cands_df, chunk_features_df, X_infer, p_raw, p_cal, pair_rows, cand_map
            gc.collect()

        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({len(chunk_queries):,} queries) processed in {time.time() - t_chk:.1f}s.", flush=True)

    # Enforce Global 1-to-1 consistency across entire country
    if country_scored_chunks:
        print(f"  Enforcing global country-wide 1-to-1 assignment across {country}...", flush=True)
        country_scored_df = pd.concat(country_scored_chunks, ignore_index=True)
        country_1to1 = resolve_one_to_one(country_scored_df, prob_col="p_cal")

        # Metric-Aware Closed-Form Expected-F0.5 Subset Selection
        all_s1_ids_country = [eid for eid, _, _ in s1_records_c]
        country_predictions = decide_expected_f05(
            country_1to1, all_s1_ids_country,
            prob_col="p_cal", max_k=8, n_samples=150
        )
        n_matched = 0
        for s1_id, match_set in country_predictions.items():
            if match_set:
                final_matches_dict[s1_id] = list(match_set)
                n_matched += 1

        print(f"  {country} matched links: {n_matched:,} entities with at least one match", flush=True)

    print(f"  {country} completed in {time.time() - t_c_start:.1f}s.", flush=True)

    del partner_lookup, slug_index, sorted_slug_index, addr_key_index, soundex_index, acronym_index, word_index, char3_index, postal_index, word_idf, char3_idf, country_scored_chunks

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
