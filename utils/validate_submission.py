#!/usr/bin/env python3
"""
Official Submission Validator for Amazon ML Challenge: Business Entity Resolution.
Stdlib only, zero external dependencies.
Checks matching_results.tsv and candidate_pairs.tsv against test source files.
"""

import sys
import os
import argparse
import csv

def validate(matching_file, candidate_file, test_dir):
    issues = []
    
    # 1. Check test source files exist
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    for f, name in [(s1_file, "test_source1.tsv"), (s2_file, "test_source2.tsv"), (s3_file, "test_source3.tsv")]:
        if not os.path.exists(f):
            issues.append(f"Test file missing: {f}")
            
    if issues:
        print_issues(issues)
        return False

    # Read test source IDs
    expected_s1_ids = []
    s1_id_set = set()
    with open(s1_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "entity_id" not in reader.fieldnames:
            issues.append(f"Column 'entity_id' not found in {s1_file}")
            print_issues(issues)
            return False
        for row in reader:
            eid = row["entity_id"].strip()
            if eid:
                expected_s1_ids.append(eid)
                s1_id_set.add(eid)

    valid_s2_ids = set()
    with open(s2_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                valid_s2_ids.add(eid)

    valid_s3_ids = set()
    with open(s3_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                valid_s3_ids.add(eid)

    valid_partner_ids = valid_s2_ids | valid_s3_ids

    # 2. Validate matching_results.tsv
    if not os.path.exists(matching_file):
        issues.append(f"Matching file does not exist: {matching_file}")
        print_issues(issues)
        return False

    matches_by_s1 = {}
    with open(matching_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            issues.append(f"Invalid header in {matching_file}. Expected 2 tab-separated columns.")
        elif header[0].strip() != "source1_entity_id" or header[1].strip() != "matched_entity_ids":
            issues.append(f"Header in {matching_file} must be 'source1_entity_id\\tmatched_entity_ids', got {header}")

        seen_s1_matching = set()
        for line_num, row in enumerate(reader, start=2):
            if len(row) == 0:
                continue
            s1_id = row[0].strip()
            matched_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in seen_s1_matching:
                issues.append(f"Duplicate source1_entity_id in {matching_file}: '{s1_id}' at line {line_num}")
            seen_s1_matching.add(s1_id)

            if s1_id not in s1_id_set:
                issues.append(f"Unknown source1_entity_id in {matching_file}: '{s1_id}' not in test_source1.tsv")

            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()] if matched_str else []
            # Check for duplicate IDs in list
            if len(matched_ids) != len(set(matched_ids)):
                issues.append(f"Duplicate entity IDs in matched_entity_ids for '{s1_id}' at line {line_num}: {matched_ids}")

            for mid in matched_ids:
                if mid.startswith("S1-") or mid in s1_id_set:
                    issues.append(f"Self-match to Source 1 ID '{mid}' for '{s1_id}' in {matching_file} (forbidden)")
                elif mid not in valid_partner_ids:
                    issues.append(f"Invalid partner ID '{mid}' for '{s1_id}' in {matching_file} (not in test S2/S3)")

            matches_by_s1[s1_id] = set(matched_ids)

    # Check that every test S1 entity is present in matching_results.tsv
    missing_in_matching = s1_id_set - set(matches_by_s1.keys())
    if missing_in_matching:
        issues.append(f"{len(missing_in_matching)} Source 1 entities from test_source1.tsv are missing in {matching_file}. Examples: {list(missing_in_matching)[:5]}")

    # 3. Validate candidate_pairs.tsv
    if not os.path.exists(candidate_file):
        issues.append(f"Candidate file does not exist: {candidate_file}")
        print_issues(issues)
        return False

    candidates_by_s1 = {}
    with open(candidate_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            issues.append(f"Invalid header in {candidate_file}. Expected 2 tab-separated columns.")
        elif header[0].strip() != "source1_entity_id" or header[1].strip() != "candidate_entity_ids":
            issues.append(f"Header in {candidate_file} must be 'source1_entity_id\\tcandidate_entity_ids', got {header}")

        seen_s1_candidate = set()
        for line_num, row in enumerate(reader, start=2):
            if len(row) == 0:
                continue
            s1_id = row[0].strip()
            cand_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in seen_s1_candidate:
                issues.append(f"Duplicate source1_entity_id in {candidate_file}: '{s1_id}' at line {line_num}")
            seen_s1_candidate.add(s1_id)

            if s1_id not in s1_id_set:
                issues.append(f"Unknown source1_entity_id in {candidate_file}: '{s1_id}' not in test_source1.tsv")

            cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []
            if len(cand_ids) != len(set(cand_ids)):
                issues.append(f"Duplicate entity IDs in candidate_entity_ids for '{s1_id}' at line {line_num}: {cand_ids}")

            for cid in cand_ids:
                if cid.startswith("S1-") or cid in s1_id_set:
                    issues.append(f"Self-match candidate to Source 1 ID '{cid}' for '{s1_id}' in {candidate_file}")
                elif cid not in valid_partner_ids:
                    issues.append(f"Invalid candidate ID '{cid}' for '{s1_id}' in {candidate_file} (not in test S2/S3)")

            candidates_by_s1[s1_id] = set(cand_ids)

    missing_in_candidates = s1_id_set - set(candidates_by_s1.keys())
    if missing_in_candidates:
        issues.append(f"{len(missing_in_candidates)} Source 1 entities from test_source1.tsv are missing in {candidate_file}. Examples: {list(missing_in_candidates)[:5]}")

    # 4. Check subset condition: matching_results must be a subset of candidate_pairs
    for s1_id, match_set in matches_by_s1.items():
        cand_set = candidates_by_s1.get(s1_id, set())
        diff = match_set - cand_set
        if diff:
            issues.append(f"For '{s1_id}', matched IDs {diff} are not in candidate set (violates matches ⊆ candidates)")

    if issues:
        print_issues(issues)
        return False

    print("PASS")
    return True

def print_issues(issues):
    print(f"Validation failed with {len(issues)} issue(s):")
    for idx, issue in enumerate(issues[:50], 1):
        print(f"  {idx}. {issue}")
    if len(issues) > 50:
        print(f"  ... and {len(issues) - 50} more issues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official submission validator for Business Entity Resolution")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
