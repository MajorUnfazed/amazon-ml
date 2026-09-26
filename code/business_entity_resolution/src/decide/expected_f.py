"""Metric-aware Expected-F0.5 set selection layer for LinkSure."""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

def select_expected_f05_subset(
    candidates: List[str],
    probs: List[float],
    max_k: int = 8,
    n_samples: int = 250,
    random_state: int = 42
) -> Set[str]:
    """
    Selects the candidate subset (including empty set) that maximizes
    expected macro F0.5 for a single S1 entity.
    """
    if not candidates:
        return set()

    # Sort candidates descending by calibrated probability
    sort_idx = np.argsort(-np.array(probs))
    sorted_cands = [candidates[i] for i in sort_idx]
    sorted_probs = np.array([probs[i] for i in sort_idx], dtype=float).clip(0.0, 1.0)

    M = len(sorted_probs)
    K = min(M, max_k)

    # Expected score of k = 0 (empty set prediction)
    # Score is 1.0 if entity has 0 true matches, else 0.0
    # P(|T| = 0) = prod(1 - p_i)
    prob_no_matches = float(np.prod(1.0 - sorted_probs))
    expected_scores = [prob_no_matches]

    # Vectorized Monte Carlo evaluation for k >= 1
    rng = np.random.RandomState(random_state)
    U = rng.uniform(0.0, 1.0, size=(n_samples, M))
    T_draws = (U < sorted_probs).astype(float)  # Shape (n_samples, M)
    T_totals = np.sum(T_draws, axis=1)          # Shape (n_samples,)

    for k in range(1, K + 1):
        tp_k = np.sum(T_draws[:, :k], axis=1)    # Shape (n_samples,)
        scores_k = np.zeros(n_samples, dtype=float)

        # Non-zero TP and non-empty true set
        valid_mask = (tp_k > 0) & (T_totals > 0)
        if np.any(valid_mask):
            scores_k[valid_mask] = (1.25 * tp_k[valid_mask]) / (0.25 * T_totals[valid_mask] + float(k))

        expected_scores.append(float(np.mean(scores_k)))

    best_k = int(np.argmax(expected_scores))
    if best_k == 0:
        return set()
    return set(sorted_cands[:best_k])

def decide_expected_f05(
    scored_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    prob_col: str = "p_cal",
    max_k: int = 8,
    n_samples: int = 150
) -> Dict[str, Set[str]]:
    """
    Applies Expected-F0.5 set selection per Source 1 entity across the dataset.
    Vectorized dictionary grouping for maximum throughput on multi-million row tables.
    """
    predictions = {s1: set() for s1 in all_s1_ids}
    if len(scored_pairs_df) == 0:
        return predictions

    from collections import defaultdict
    cand_map = defaultdict(list)
    prob_map = defaultdict(list)

    s1_vals = scored_pairs_df["source1_entity_id"].values
    p_vals = scored_pairs_df["partner_entity_id"].values
    pr_vals = scored_pairs_df[prob_col].values

    for s1, p, pr in zip(s1_vals, p_vals, pr_vals):
        cand_map[s1].append(p)
        prob_map[s1].append(float(pr))

    for s1_id, cands in cand_map.items():
        probs = prob_map[s1_id]
        pred_set = select_expected_f05_subset(
            cands, probs, max_k=max_k, n_samples=n_samples
        )
        if pred_set:
            predictions[s1_id] = pred_set

    return predictions

def decide_global_threshold(
    scored_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    prob_col: str = "p_cal",
    threshold: float = 0.50
) -> Dict[str, Set[str]]:
    """
    Baseline decision: select all pairs where probability >= threshold.
    """
    predictions = {s1: set() for s1 in all_s1_ids}
    if len(scored_pairs_df) == 0:
        return predictions

    passing = scored_pairs_df[scored_pairs_df[prob_col] >= threshold]
    grouped = passing.groupby("source1_entity_id")["partner_entity_id"].apply(set)
    for s1_id, matches in grouped.items():
        predictions[s1_id] = matches
    return predictions

def decide_two_stage_threshold(
    scored_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    prob_col: str = "p_cal",
    t_first: float = 0.45,
    t_extra: float = 0.70
) -> Dict[str, Set[str]]:
    """
    Two-stage decision: predict top-1 if prob >= t_first, plus extra candidates if prob >= t_extra.
    """
    predictions = {s1: set() for s1 in all_s1_ids}
    if len(scored_pairs_df) == 0:
        return predictions

    grouped = scored_pairs_df.groupby("source1_entity_id")
    for s1_id, group in grouped:
        sorted_g = group.sort_values(by=prob_col, ascending=False)
        cands = sorted_g["partner_entity_id"].tolist()
        probs = sorted_g[prob_col].tolist()

        selected = set()
        if probs and probs[0] >= t_first:
            selected.add(cands[0])
            for c, p in zip(cands[1:], probs[1:]):
                if p >= t_extra:
                    selected.add(c)

        predictions[s1_id] = selected

    return predictions
