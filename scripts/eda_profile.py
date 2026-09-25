"""Profile real competition dataset: answers A1, A2, singleton rates, and distributions."""

import os
import sys
import time
from collections import Counter

def profile_data():
    t0 = time.time()
    train_dir = os.path.join("dataset", "train")
    test_dir = os.path.join("dataset", "test")

    print("=" * 70)
    print("LinkSure: Real Competition Data Profiling & Question Answering")
    print("=" * 70)

    # 1. Profile train_source1.tsv
    print("\n[1/5] Profiling train_source1.tsv...")
    s1_countries = Counter()
    s1_ids = set()
    s1_country_map = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
        header = next(f).strip().split("\t")
        print("  Header:", header)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_ids.add(eid)
                s1_countries[country] += 1
                s1_country_map[eid] = country

    print(f"  Total S1 Entities: {len(s1_ids):,}")
    print("  S1 Country Breakdown:")
    for c, cnt in s1_countries.most_common():
        print(f"    - {c}: {cnt:,} ({cnt/len(s1_ids):.2%})")

    # 2. Profile train_ground_truth.tsv
    print("\n[2/5] Profiling train_ground_truth.tsv (A1, Singleton rate, match counts)...")
    gt_entities = 0
    singletons = 0
    singletons_per_country = Counter()
    matches_per_entity_dist = Counter()
    partner_claims = {} # partner_id -> list of s1_ids
    gt_pairs_by_s1 = {}

    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
        header = next(f).strip().split("\t")
        print("  Header:", header)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            gt_entities += 1
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""

            if matched_str:
                partners = [p.strip() for p in matched_str.split(",") if p.strip()]
            else:
                partners = []

            n_matches = len(partners)
            matches_per_entity_dist[n_matches] += 1
            gt_pairs_by_s1[s1_id] = partners

            s1_country = s1_country_map.get(s1_id, "Unknown")
            if n_matches == 0:
                singletons += 1
                singletons_per_country[s1_country] += 1

            for p in partners:
                partner_claims.setdefault(p, []).append(s1_id)

    print(f"  Total GT Entities: {gt_entities:,}")
    print(f"  Overall Singletons: {singletons:,} ({singletons / gt_entities:.2%})")
    print("  Singletons by Country:")
    for c, s_cnt in singletons_per_country.items():
        total_c = s1_countries.get(c, 1)
        print(f"    - {c}: {s_cnt:,} singletons ({s_cnt / total_c:.2%} of {c})")

    print("\n  Matches per S1 Entity Distribution:")
    for num_m in sorted(matches_per_entity_dist.keys())[:10]:
        cnt = matches_per_entity_dist[num_m]
        print(f"    - {num_m} matches: {cnt:,} entities ({cnt / gt_entities:.2%})")
    if len(matches_per_entity_dist) > 10:
        over_10 = sum(cnt for m, cnt in matches_per_entity_dist.items() if m >= 10)
        print(f"    - 10+ matches: {over_10:,} entities ({over_10 / gt_entities:.2%})")

    # Question A1: Can one S2/S3 record match more than one S1?
    print("\n  Checking Question A1 (One-to-One Consistency)...")
    multi_claimed = {p: s1_list for p, s1_list in partner_claims.items() if len(s1_list) > 1}
    print(f"  Total Unique Partner IDs in GT: {len(partner_claims):,}")
    print(f"  Partners claimed by >1 S1 entity: {len(multi_claimed):,}")
    if multi_claimed:
        print("    Examples of multi-claimed partners:")
        for p, s1_list in list(multi_claimed.items())[:5]:
            print(f"      {p} claimed by: {s1_list}")
    else:
        print("    [CONFIRMED] Zero partner IDs are shared by multiple S1 entities! One-to-one constraint holds 100%!")

    # 3. Check Question A2: Country Agreement
    print("\n[3/5] Checking Question A2 (Country Agreement between S1 and S2/S3)...")
    # Sample 100k partners from S2 and S3 to check country consistency
    partner_country_map = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        fp = os.path.join(train_dir, src_file)
        print(f"  Reading country labels from {src_file}...")
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    pid, country = parts[0], parts[3]
                    if pid in partner_claims:
                        partner_country_map[pid] = country

    print(f"  Loaded country labels for {len(partner_country_map):,} ground-truth partners.")
    country_mismatches = 0
    total_matched_links = 0
    for s1_id, partners in gt_pairs_by_s1.items():
        s1_c = s1_country_map.get(s1_id)
        for p in partners:
            total_matched_links += 1
            p_c = partner_country_map.get(p)
            if p_c and s1_c and p_c != s1_c:
                country_mismatches += 1

    print(f"  Total GT Links evaluated: {total_matched_links:,}")
    print(f"  Country Mismatches: {country_mismatches:,}")
    if total_matched_links > 0:
        agreement_rate = (1.0 - country_mismatches / total_matched_links) * 100.0
        print(f"  Country Agreement Rate: {agreement_rate:.4f}%")

    # 4. Profile test_source1.tsv (including France!)
    print("\n[4/5] Profiling test_source1.tsv (including France!)...")
    test_countries = Counter()
    test_s1_count = 0
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
        header = next(f).strip().split("\t")
        print("  Header:", header)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                test_s1_count += 1
                test_countries[parts[3]] += 1

    print(f"  Total Test S1 Entities: {test_s1_count:,}")
    print("  Test Country Breakdown:")
    for c, cnt in test_countries.most_common():
        print(f"    - {c}: {cnt:,} ({cnt / test_s1_count:.2%})")

    # 5. Profile partner counts in S2 and S3
    print("\n[5/5] Counting total partner pools...")
    s2_train_count = sum(1 for _ in open(os.path.join(train_dir, "train_source2.tsv"), "rb")) - 1
    s3_train_count = sum(1 for _ in open(os.path.join(train_dir, "train_source3.tsv"), "rb")) - 1
    s2_test_count = sum(1 for _ in open(os.path.join(test_dir, "test_source2.tsv"), "rb")) - 1
    s3_test_count = sum(1 for _ in open(os.path.join(test_dir, "test_source3.tsv"), "rb")) - 1

    print(f"  Train: Source 2 = {s2_train_count:,}, Source 3 = {s3_train_count:,} (Total: {s2_train_count + s3_train_count:,})")
    print(f"  Test:  Source 2 = {s2_test_count:,}, Source 3 = {s3_test_count:,} (Total: {s2_test_count + s3_test_count:,})")

    elapsed = time.time() - t0
    print(f"\nProfiling completed in {elapsed:.1f}s.")
    print("=" * 70)

if __name__ == "__main__":
    profile_data()
