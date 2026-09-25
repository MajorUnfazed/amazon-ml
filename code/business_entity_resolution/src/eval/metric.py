"""Official Metric Implementation: Macro-averaged F_0.5 Score.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
     = (1.25 * TP) / (0.25 * |T| + |P|)

Singleton handling:
- True empty, Pred empty => 1.0
- True empty, Pred non-empty => 0.0
- True non-empty, Pred empty => 0.0
"""

from typing import Dict, Set, List, Tuple, Any
import numpy as np

def compute_per_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (F0.5, precision, recall) for a single Source 1 entity.
    """
    len_true = len(true_ids)
    len_pred = len(pred_ids)

    # Both empty: correctly identified singleton
    if len_true == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0

    # True singleton but predicted matches: false merge
    if len_true == 0 and len_pred > 0:
        return 0.0, 0.0, 1.0  # Precision 0.0, recall 1.0 (empty target)

    # True non-singleton but predicted empty: missed match
    if len_true > 0 and len_pred == 0:
        return 0.0, 1.0, 0.0  # Precision 1.0 (no false alarms), recall 0.0

    # Both non-empty
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_pred
    recall = tp / len_true

    # Exact formula from problem statement
    f05 = (1.25 * tp) / (0.25 * len_true + len_pred)
    return float(f05), precision, recall

def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground truth.
    All entities in ground_truth must be evaluated.
    """
    if not ground_truth:
        return 0.0

    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05, _, _ = compute_per_entity_f05(true_set, pred_set)
        scores.append(f05)

    return float(np.mean(scores))

def evaluate_predictions(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Comprehensive evaluation breakdown:
    - macro_f05: Main competition metric
    - singleton_accuracy: Accuracy on true empty entities
    - non_singleton_f05: Macro F0.5 on entities with true matches
    - mean_precision: Macro precision
    - mean_recall: Macro recall
    - total_entities: Total S1 entities
    - singleton_ratio: Fraction of entities that are singletons
    """
    total = len(ground_truth)
    if total == 0:
        return {}

    f05_list = []
    prec_list = []
    rec_list = []

    singleton_correct = 0
    singleton_total = 0
    non_singleton_f05 = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05, p, r = compute_per_entity_f05(true_set, pred_set)
        f05_list.append(f05)
        prec_list.append(p)
        rec_list.append(r)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_f05.append(f05)

    return {
        "macro_f05": float(np.mean(f05_list)),
        "mean_precision": float(np.mean(prec_list)),
        "mean_recall": float(np.mean(rec_list)),
        "singleton_accuracy": float(singleton_correct / singleton_total) if singleton_total > 0 else 0.0,
        "singleton_total": singleton_total,
        "non_singleton_f05": float(np.mean(non_singleton_f05)) if non_singleton_f05 else 0.0,
        "non_singleton_total": len(non_singleton_f05),
        "total_entities": total,
        "singleton_ratio": float(singleton_total / total)
    }

def compute_blocking_metrics(ground_truth: Dict[str, Set[str]], candidates: Dict[str, Set[str]], total_pool_size: int = 0) -> Dict[str, float]:
    """
    Computes blocking audit metrics:
    - recall_ceiling: Fraction of true matching pairs retained in candidate sets
    - avg_candidates_per_s1: Average candidate count per S1 entity
    - reduction_ratio: 1 - (total candidate pairs / total possible pairs)
    """
    total_true_pairs = 0
    found_true_pairs = 0
    total_candidate_pairs = 0

    for s1_id, true_set in ground_truth.items():
        total_true_pairs += len(true_set)
        cand_set = candidates.get(s1_id, set())
        found_true_pairs += len(true_set & cand_set)
        total_candidate_pairs += len(cand_set)

    n_s1 = len(ground_truth)
    recall_ceiling = (found_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_cands = (total_candidate_pairs / n_s1) if n_s1 > 0 else 0.0

    metrics = {
        "recall_ceiling": float(recall_ceiling),
        "found_true_pairs": found_true_pairs,
        "total_true_pairs": total_true_pairs,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_s1": float(avg_cands)
    }

    if total_pool_size > 0 and n_s1 > 0:
        total_possible = n_s1 * total_pool_size
        metrics["reduction_ratio"] = float(1.0 - (total_candidate_pairs / total_possible))

    return metrics
