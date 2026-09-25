"""Unit tests for LinkSure Metric Implementation."""

import os
import sys
import numpy as np

# Ensure code/business_entity_resolution/src is in sys.path
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from eval.metric import (
    compute_per_entity_f05,
    compute_macro_f05,
    evaluate_predictions,
    compute_blocking_metrics
)

def test_pdf_example():
    """
    Problem statement PDF Page 6 Example:
    - Model predicts: [S2-00047, S2-00193, S3-00812] (|P| = 3)
    - Ground truth:   [S2-00047, S3-00812] (|T| = 2)
    - Precision = 2/3, Recall = 2/2 = 1.0
    - F_0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
    """
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}

    f05, prec, rec = compute_per_entity_f05(true_set, pred_set)
    assert np.isclose(f05, 5.0 / 7.0, atol=1e-5), f"Expected 5/7 (~0.7142857), got {f05}"
    assert np.isclose(round(f05, 3), 0.714)
    assert np.isclose(prec, 2.0 / 3.0)
    assert np.isclose(rec, 1.0)

def test_singletons():
    # Correct singleton: true empty, pred empty => 1.0
    f05_empty_empty, _, _ = compute_per_entity_f05(set(), set())
    assert f05_empty_empty == 1.0

    # False merge on singleton: true empty, pred non-empty => 0.0
    f05_empty_pred, _, _ = compute_per_entity_f05(set(), {"S2-00001"})
    assert f05_empty_pred == 0.0

    # Missed match: true non-empty, pred empty => 0.0
    f05_true_empty, _, _ = compute_per_entity_f05({"S2-00001"}, set())
    assert f05_true_empty == 0.0

def test_gt_against_itself():
    # Scoring GT against itself must return exactly 1.0
    gt = {
        "S1-001": {"S2-001", "S3-002"},
        "S1-002": {"S2-003"},
        "S1-003": set(),  # singleton
        "S1-004": set()   # singleton
    }
    score = compute_macro_f05(gt, gt)
    assert score == 1.0

def test_all_empty_equals_singleton_fraction():
    # An all-empty submission's score must equal the singleton fraction
    gt = {
        "S1-001": {"S2-001"},
        "S1-002": {"S2-002"},
        "S1-003": set(),
        "S1-004": set(),
        "S1-005": set()
    }
    empty_preds = {s1: set() for s1 in gt}
    score = compute_macro_f05(gt, empty_preds)
    # 3 singletons out of 5 entities => 3/5 = 0.60
    assert np.isclose(score, 0.60)

    report = evaluate_predictions(gt, empty_preds)
    assert report["macro_f05"] == 0.60
    assert report["singleton_accuracy"] == 1.0
    assert report["singleton_ratio"] == 0.60

def test_blocking_metrics():
    gt = {
        "S1-001": {"S2-001", "S2-002"},
        "S1-002": {"S3-001"},
        "S1-003": set()
    }
    candidates = {
        "S1-001": {"S2-001", "S2-999"},  # 1 hit, 1 miss
        "S1-002": {"S3-001"},            # 1 hit
        "S1-003": {"S2-123"}             # 0 hit
    }
    # Total true pairs: 2 + 1 = 3. Found: 1 + 1 = 2. Recall ceiling = 2/3.
    metrics = compute_blocking_metrics(gt, candidates, total_pool_size=100)
    assert np.isclose(metrics["recall_ceiling"], 2.0 / 3.0)
    assert metrics["found_true_pairs"] == 2
    assert metrics["total_true_pairs"] == 3
    assert metrics["avg_candidates_per_s1"] == 4.0 / 3.0
    assert metrics["reduction_ratio"] > 0.98

if __name__ == "__main__":
    test_pdf_example()
    test_singletons()
    test_gt_against_itself()
    test_all_empty_equals_singleton_fraction()
    test_blocking_metrics()
    print("All metric tests passed successfully!")
