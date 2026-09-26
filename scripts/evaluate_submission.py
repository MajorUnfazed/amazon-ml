#!/usr/bin/env python3
"""
Official Amazon ML Challenge Metric Evaluator: Macro-Averaged F0.5.

Computes exact competition evaluation metrics:
1. Macro F0.5 (exact official per-entity formula: 5*TP / (|Y| + 4*|Y_hat|))
2. Singleton Accuracy (entities with 0 true matches)
3. Non-Singleton F0.5 (entities with >= 1 true matches)
4. Macro Precision & Macro Recall
5. Secondary Audit: Average candidate set size (C_bar) & Blocking Recall Ceiling

Usage:
    python scripts/evaluate_submission.py --ground-truth <path> --predictions <path> [--candidates <path>]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple

def parse_tsv_dict(filepath: str, key_col_idx: int = 0, val_col_idx: int = 1) -> Dict[str, Set[str]]:
    """Reads a TSV file into a dictionary mapping entity_id -> set of match/candidate IDs."""
    mapping = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            k = parts[key_col_idx].strip()
            if len(parts) > val_col_idx and parts[val_col_idx].strip():
                v = set(x.strip() for x in parts[val_col_idx].split(",") if x.strip())
            else:
                v = set()
            mapping[k] = v
    return mapping

def compute_entity_f05(y_true: Set[str], y_pred: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (F0.5, precision, recall) for a single Source 1 entity
    strictly according to the official competition specification:
        F0.5 = 5 * TP / (|Y| + 4 * |Y_hat|)
    """
    len_true = len(y_true)
    len_pred = len(y_pred)

    # Case 1: True Singleton (Correct non-match)
    if len_true == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0

    # Case 2: False Positive Singleton (Spurious match on empty target)
    if len_true == 0 and len_pred > 0:
        return 0.0, 0.0, 1.0

    # Case 3: Complete Miss (False negative on non-empty target)
    if len_true > 0 and len_pred == 0:
        return 0.0, 1.0, 0.0

    # Case 4: Both non-empty
    tp = len(y_true & y_pred)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_pred
    recall = tp / len_true

    # Exact simplified competition formula: 5 * TP / (|Y| + 4 * |Y_hat|)
    f05 = (5.0 * tp) / (float(len_true) + 4.0 * float(len_pred))
    return float(f05), precision, recall

def evaluate_predictions(
    ground_truth_path: str,
    predictions_path: str,
    candidates_path: str = None
) -> dict:
    """Evaluates prediction and candidate files against ground truth."""
    print("=" * 70)
    print("Amazon ML Challenge: Official Metric Evaluation Engine")
    print("=" * 70)

    print(f"Loading Ground Truth: {ground_truth_path}...")
    gt = parse_tsv_dict(ground_truth_path)
    total_entities = len(gt)
    print(f"  Total Ground Truth entities: {total_entities:,}")

    print(f"Loading Predictions:  {predictions_path}...")
    preds = parse_tsv_dict(predictions_path)
    print(f"  Total Prediction entries:  {len(preds):,}")

    scores = []
    precisions = []
    recalls = []

    singleton_total = 0
    singleton_correct = 0
    non_singleton_scores = []

    for s1_id, y_true in gt.items():
        y_pred = preds.get(s1_id, set())
        f05, p, r = compute_entity_f05(y_true, y_pred)

        scores.append(f05)
        precisions.append(p)
        recalls.append(r)

        if len(y_true) == 0:
            singleton_total += 1
            if len(y_pred) == 0:
                singleton_correct += 1
        else:
            non_singleton_scores.append(f05)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(singleton_correct / singleton_total) if singleton_total > 0 else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    print("\n" + "-" * 70)
    print(f"  PRIMARY LEADERBOARD METRIC:")
    print(f"  >>> MACRO F0.5 SCORE : {macro_f05:.4f} <<<")
    print("-" * 70)
    print(f"  Precision (Macro)    : {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"  Recall (Macro)       : {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"  Singleton Accuracy   : {singleton_acc:.4f} ({singleton_acc*100:.2f}%) [{singleton_correct:,} / {singleton_total:,}]")
    print(f"  Non-Singleton F0.5   : {non_singleton_f05:.4f} ({len(non_singleton_scores):,} entities)")
    print(f"  Evaluated Entities   : {len(scores):,}")

    # Secondary Candidate Audit if candidates_path is provided
    if candidates_path and os.path.exists(candidates_path):
        print("\n" + "-" * 70)
        print("  SECONDARY RANKING AUDIT (Candidate Generation):")
        print(f"  Loading Candidates : {candidates_path}...")
        cands = parse_tsv_dict(candidates_path)

        cand_counts = [len(cands.get(s1_id, set())) for s1_id in gt]
        avg_cand_size = float(np.mean(cand_counts)) if cand_counts else 0.0

        true_pairs = sum(len(y) for y in gt.values())
        retained_pairs = sum(len(gt[s1_id] & cands.get(s1_id, set())) for s1_id in gt)
        recall_ceiling = (retained_pairs / true_pairs) if true_pairs > 0 else 1.0

        # Consistency check: every predicted match must be in candidates
        inconsistent_count = 0
        for s1_id, y_pred in preds.items():
            cand_set = cands.get(s1_id, set())
            if not y_pred.issubset(cand_set):
                inconsistent_count += 1

        print(f"  Average Candidate Set Size (C_bar) : {avg_cand_size:.2f} candidates/entity")
        print(f"  Blocking Recall Ceiling             : {recall_ceiling:.2%} ({retained_pairs:,} / {true_pairs:,} true links retained)")
        print(f"  Candidate Consistency Status        : {'PASS (100% Consistent)' if inconsistent_count == 0 else f'FAIL ({inconsistent_count} inconsistent entities)'}")

    print("=" * 70)

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "total_evaluated": len(scores)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction files with official Amazon ML Challenge Macro F0.5")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV file")
    parser.add_argument("--predictions", "-p", required=True, help="Path to matching_results.tsv file")
    parser.add_argument("--candidates", "-c", default=None, help="Optional path to candidate_pairs.tsv file for secondary audit")
    args = parser.parse_args()

    evaluate_predictions(args.ground_truth, args.predictions, args.candidates)
