"""One-command entry point for LinkSure Business Entity Resolution Pipeline."""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure src directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from config import LinkSureConfig, DEFAULT_CONFIG
from data.loader import load_dataset, profile_dataset, parse_ground_truth_dict
from normalize.text import normalize_record
from blocking.candidate_manager import CandidateManager
from features.builder import FeatureBuilder
from model.matcher import EntityResolutionMatcher
from decide.one_to_one import resolve_one_to_one
from decide.expected_f import (
    decide_expected_f05,
    decide_global_threshold,
    decide_two_stage_threshold
)
from eval.metric import evaluate_predictions, compute_blocking_metrics
from pipeline_io.writer import write_tsv_submission, run_submission_validator

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies normalization engine to each record in DataFrame."""
    norm_records = []
    for _, row in df.iterrows():
        norm_dict = normalize_record(
            name=row["business_name"],
            address=row["business_address"],
            country=row["country"]
        )
        norm_dict["entity_id"] = row["entity_id"]
        norm_dict["country_raw"] = row["country"]
        norm_records.append(norm_dict)

    return pd.DataFrame(norm_records)

def run_pipeline(
    data_dir: str,
    output_dir: str,
    mode: str = "full",
    config: LinkSureConfig = DEFAULT_CONFIG
):
    print("=" * 70)
    print(f"LinkSure Business Entity Resolution Pipeline [Mode: {mode.upper()}]")
    print(f"Data Directory:   {data_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.artifacts_dir, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/7] Loading datasets...")
    train_s1, train_s2, train_s3, train_gt = load_dataset(data_dir, split="train")
    test_s1, test_s2, test_s3, _ = load_dataset(data_dir, split="test")

    train_partner = pd.concat([train_s2, train_s3], ignore_index=True)
    test_partner = pd.concat([test_s2, test_s3], ignore_index=True)

    gt_dict = parse_ground_truth_dict(train_gt) if train_gt is not None else {}

    print(f"  Train: S1={len(train_s1):,}, Partners(S2+S3)={len(train_partner):,}, GT Entities={len(gt_dict):,}")
    print(f"  Test:  S1={len(test_s1):,}, Partners(S2+S3)={len(test_partner):,}")

    profile = profile_dataset(train_s1, train_s2, train_s3, train_gt)
    print(f"  Train Profile: Singleton Rate = {profile.get('singleton_rate', 0.0):.2%}, One-to-One holds = {profile.get('a1_one_to_one_holds', False)}")

    # Handle Trivial Mode
    if mode == "trivial":
        print("\n[Trivial Mode] Generating all-empty baseline predictions...")
        empty_preds = {s1: set() for s1 in test_s1["entity_id"]}
        match_out = os.path.join(output_dir, "matching_results.tsv")
        cand_out = os.path.join(output_dir, "candidate_pairs.tsv")

        write_tsv_submission(match_out, empty_preds, test_s1["entity_id"].tolist(), "source1_entity_id", "matched_entity_ids")
        write_tsv_submission(cand_out, empty_preds, test_s1["entity_id"].tolist(), "source1_entity_id", "candidate_entity_ids")

        print("  Running validator...")
        val_pass, val_msg = run_submission_validator(match_out, cand_out, os.path.join(data_dir, "test"))
        print(f"  Validator output:\n{val_msg}")
        return

    # 2. Normalize
    print("\n[Step 2/7] Normalizing entity names, addresses, and postcodes...")
    norm_train_s1 = normalize_dataframe(train_s1)
    norm_train_partner = normalize_dataframe(train_partner)

    norm_test_s1 = normalize_dataframe(test_s1)
    norm_test_partner = normalize_dataframe(test_partner)
    print("  Normalization complete.")

    # 3. Blocking / Candidate Generation
    print("\n[Step 3/7] Running multi-retriever candidate generation...")
    blocker = CandidateManager(
        top_k_name=config.top_k_name_tfidf,
        top_k_name_addr=config.top_k_name_addr_tfidf,
        top_k_addr=config.top_k_addr_tfidf,
        max_candidates_per_s1=config.max_candidates_per_s1,
        scope_by_country=config.scope_by_country
    )

    train_cands_df = blocker.generate_candidates(norm_train_s1, norm_train_partner)
    test_cands_df = blocker.generate_candidates(norm_test_s1, norm_test_partner)

    print(f"  Generated {len(train_cands_df):,} train candidate pairs and {len(test_cands_df):,} test candidate pairs.")

    # Audit blocking on train
    train_cands_dict = blocker.to_candidate_dict(train_cands_df, train_s1["entity_id"].tolist())
    test_cands_dict = blocker.to_candidate_dict(test_cands_df, test_s1["entity_id"].tolist())

    if gt_dict:
        blocking_stats = compute_blocking_metrics(gt_dict, train_cands_dict, total_pool_size=len(train_partner))
        print(f"  Blocking Audit: Recall Ceiling = {blocking_stats['recall_ceiling']:.2%}, Avg Candidates/S1 = {blocking_stats['avg_candidates_per_s1']:.1f}, Reduction Ratio = {blocking_stats.get('reduction_ratio', 0.0):.4%}")

    # Handle Rule Baseline Mode
    if mode == "baseline":
        print("\n[Baseline Mode] Applying rule threshold on name TF-IDF similarity...")
        baseline_preds = {}
        for s1_id in test_s1["entity_id"]:
            sub = test_cands_df[test_cands_df["source1_entity_id"] == s1_id]
            matches = set(sub[sub["score_name_tfidf"] >= 0.70]["partner_entity_id"])
            baseline_preds[s1_id] = matches

        match_out = os.path.join(output_dir, "matching_results.tsv")
        cand_out = os.path.join(output_dir, "candidate_pairs.tsv")

        write_tsv_submission(match_out, baseline_preds, test_s1["entity_id"].tolist(), "source1_entity_id", "matched_entity_ids")
        write_tsv_submission(cand_out, test_cands_dict, test_s1["entity_id"].tolist(), "source1_entity_id", "candidate_entity_ids")

        val_pass, val_msg = run_submission_validator(match_out, cand_out, os.path.join(data_dir, "test"))
        print(f"  Validator output:\n{val_msg}")
        return

    # 4. Feature Engineering
    print("\n[Step 4/7] Computing relational pair features...")
    fb = FeatureBuilder()
    corpus_texts = (
        norm_train_s1["name_clean"].tolist() +
        norm_train_partner["name_clean"].tolist() +
        norm_test_s1["name_clean"].tolist() +
        norm_test_partner["name_clean"].tolist()
    )
    fb.fit_idf(corpus_texts)

    train_features_df = fb.build_features(train_cands_df, norm_train_s1, norm_train_partner)
    test_features_df = fb.build_features(test_cands_df, norm_test_s1, norm_test_partner)

    # Label training pairs from Ground Truth
    gt_pairs_set = {(s1, p) for s1, plist in gt_dict.items() for p in plist}
    train_features_df["label"] = [
        1 if (s1, p) in gt_pairs_set else 0
        for s1, p in zip(train_features_df["source1_entity_id"], train_features_df["partner_entity_id"])
    ]

    feature_cols = [
        c for c in train_features_df.columns
        if c not in {"source1_entity_id", "partner_entity_id", "label"}
    ]
    print(f"  Constructed {len(feature_cols)} features per candidate pair.")

    # 5. Model Training & Calibration
    print("\n[Step 5/7] Training LightGBM matcher with 5-fold CV & Probability Calibration...")
    matcher = EntityResolutionMatcher(lgb_params=config.lgb_params)
    oof_df, feat_importances = matcher.train_cv(
        features_df=train_features_df,
        feature_cols=feature_cols,
        label_col="label",
        n_splits=config.n_splits
    )

    print("  Top 10 Feature Importances:")
    for idx, (fname, imp) in enumerate(list(feat_importances.items())[:10], 1):
        print(f"    {idx}. {fname:<30}: {imp:.1f}")

    # 6. Decision Layer Comparison on Validation (OOF)
    print("\n[Step 6/7] Evaluating decision strategies on out-of-fold predictions...")
    all_train_s1_ids = train_s1["entity_id"].tolist()

    # Baseline: Global threshold (0.50)
    pred_global = decide_global_threshold(oof_df, all_train_s1_ids, prob_col="p_cal", threshold=0.50)
    eval_global = evaluate_predictions(gt_dict, pred_global)
    print(f"  Global Threshold (0.50):           Macro F0.5 = {eval_global['macro_f05']:.4f} (Singleton Acc = {eval_global['singleton_accuracy']:.2%})")

    # Strategy 2: One-to-one + Global threshold
    oof_1to1 = resolve_one_to_one(oof_df, prob_col="p_cal")
    pred_1to1_thresh = decide_global_threshold(oof_1to1, all_train_s1_ids, prob_col="p_cal", threshold=0.45)
    eval_1to1_thresh = evaluate_predictions(gt_dict, pred_1to1_thresh)
    print(f"  1-to-1 + Threshold (0.45):         Macro F0.5 = {eval_1to1_thresh['macro_f05']:.4f} (Singleton Acc = {eval_1to1_thresh['singleton_accuracy']:.2%})")

    # Strategy 3: Full LinkSure (One-to-one + Expected-F0.5 set selection)
    pred_expected_f = decide_expected_f05(oof_1to1, all_train_s1_ids, prob_col="p_cal", max_k=config.max_k_expected_f)
    eval_expected_f = evaluate_predictions(gt_dict, pred_expected_f)
    print(f"  LinkSure (1-to-1 + Expected-F0.5): Macro F0.5 = {eval_expected_f['macro_f05']:.4f} (Singleton Acc = {eval_expected_f['singleton_accuracy']:.2%}, Non-singleton F0.5 = {eval_expected_f['non_singleton_f05']:.4f})")

    # 7. Test Inference & Final Output Generation
    print("\n[Step 7/7] Generating final predictions for test dataset...")
    all_test_s1_ids = test_s1["entity_id"].tolist()

    if len(test_features_df) > 0:
        p_raw, p_cal = matcher.predict_proba(test_features_df)
        test_scored_df = test_features_df[["source1_entity_id", "partner_entity_id"]].copy()
        test_scored_df["p_raw"] = p_raw
        test_scored_df["p_cal"] = p_cal

        # Apply LinkSure Decision Layer
        if config.enforce_one_to_one:
            test_scored_df = resolve_one_to_one(test_scored_df, prob_col="p_cal")

        final_test_predictions = decide_expected_f05(
            test_scored_df,
            all_test_s1_ids,
            prob_col="p_cal",
            max_k=config.max_k_expected_f
        )
    else:
        final_test_predictions = {s1: set() for s1 in all_test_s1_ids}

    # Write output TSVs
    match_file = os.path.join(output_dir, "matching_results.tsv")
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"  Writing matching results to {match_file}...")
    write_tsv_submission(
        match_file,
        final_test_predictions,
        all_test_s1_ids,
        id_col_name="source1_entity_id",
        list_col_name="matched_entity_ids"
    )

    print(f"  Writing candidate pairs to {cand_file}...")
    write_tsv_submission(
        cand_file,
        test_cands_dict,
        all_test_s1_ids,
        id_col_name="source1_entity_id",
        list_col_name="candidate_entity_ids"
    )

    # Official Validator Check
    print("  Validating submission files with official validator...")
    val_pass, val_msg = run_submission_validator(
        matching_tsv_path=match_file,
        candidate_tsv_path=cand_file,
        test_dir=os.path.join(data_dir, "test")
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    if val_pass:
        print(f"SUBMISSION VALIDATION: SUCCESS (PASS)")
        print(f"Completed in {elapsed:.1f}s.")
    else:
        print(f"SUBMISSION VALIDATION ISSUES:\n{val_msg}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinkSure Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output-dir", default="output", help="Path to output directory for TSVs")
    parser.add_argument("--mode", default="full", choices=["full", "baseline", "trivial"], help="Pipeline execution mode")
    args = parser.parse_args()

    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, mode=args.mode)
