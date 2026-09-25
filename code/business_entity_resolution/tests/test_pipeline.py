"""End-to-end integration test for LinkSure pipeline on synthetic data."""

import os
import sys
import shutil
import tempfile
import pandas as pd
import pytest

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from run import run_pipeline
from pipeline_io.writer import run_submission_validator

def create_synthetic_dataset(root_dir: str):
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Train S1 (US and India)
    train_s1 = pd.DataFrame([
        {"entity_id": "S1-0001", "business_name": "Apex Technology Solutions Inc.", "business_address": "100 Market St, Suite 400, San Francisco, CA 94105", "country": "US"},
        {"entity_id": "S1-0002", "business_name": "Sharma Textiles Pvt Ltd", "business_address": "45 Ring Road, Near Surat Railway Station, Surat 395002", "country": "India"},
        {"entity_id": "S1-0003", "business_name": "Lone Star Singleton Diner", "business_address": "99 Desert Hwy, Austin, TX 78701", "country": "US"},
        {"entity_id": "S1-0004", "business_name": "Global Logistics Corp dba Speedy Freight", "business_address": "12 Harbor Blvd, Boston, MA 02110", "country": "US"},
        {"entity_id": "S1-0005", "business_name": "Patel Sweets & Snacks", "business_address": "Opposite Town Hall, MG Road, Ahmedabad 380001", "country": "India"},
    ])

    # Train S2
    train_s2 = pd.DataFrame([
        {"entity_id": "S2-0001", "business_name": "Apex Tech Solutions", "business_address": "100 Market Street, Ste 400, SF, 94105", "country": "US"},
        {"entity_id": "S2-0002", "business_name": "Sharma Textiles", "business_address": "45 Ring Rd, Nr Rly Station, Surat, Gujarat 395002", "country": "India"},
        {"entity_id": "S2-0003", "business_name": "Speedy Freight", "business_address": "12 Harbor Boulevard, Boston 02110", "country": "US"},
        {"entity_id": "S2-0004", "business_name": "Unrelated Chicago Pizza Co", "business_address": "22 Michigan Ave, Chicago, IL 60601", "country": "US"},
    ])

    # Train S3
    train_s3 = pd.DataFrame([
        {"entity_id": "S3-0001", "business_name": "Apex Technology Inc", "business_address": "100 Market St, San Francisco 94105", "country": "US"},
        {"entity_id": "S3-0002", "business_name": "Patel Sweets and Snacks", "business_address": "Opp Town Hall, M.G. Road, Ahmedabad 380001", "country": "India"},
        {"entity_id": "S3-0003", "business_name": "Random Indian Bakery", "business_address": "78 Brigade Rd, Bangalore 560025", "country": "India"},
    ])

    # Train Ground Truth
    train_gt = pd.DataFrame([
        {"source1_entity_id": "S1-0001", "matched_entity_ids": "S2-0001,S3-0001"},
        {"source1_entity_id": "S1-0002", "matched_entity_ids": "S2-0002"},
        {"source1_entity_id": "S1-0003", "matched_entity_ids": ""}, # True singleton
        {"source1_entity_id": "S1-0004", "matched_entity_ids": "S2-0003"},
        {"source1_entity_id": "S1-0005", "matched_entity_ids": "S3-0002"},
    ])

    # Test S1 (US, India, and France)
    test_s1 = pd.DataFrame([
        {"entity_id": "S1-9001", "business_name": "Dubois Boulangerie S.A.R.L.", "business_address": "14 Bd. Saint-Germain, 75005 Paris", "country": "France"},
        {"entity_id": "S1-9002", "business_name": "Acme Widgets LLC", "business_address": "500 Broadway, New York, NY 10012", "country": "US"},
        {"entity_id": "S1-9003", "business_name": "Bengaluru IT Consultants", "business_address": "100 Outer Ring Rd, Marathahalli, Bangalore 560037", "country": "India"},
    ])

    # Test S2
    test_s2 = pd.DataFrame([
        {"entity_id": "S2-9001", "business_name": "Boulangerie Dubois", "business_address": "14 Boulevard Saint Germain, 75005 Paris", "country": "France"},
        {"entity_id": "S2-9002", "business_name": "Acme Widgets", "business_address": "500 Broadway St, NY 10012", "country": "US"},
    ])

    # Test S3
    test_s3 = pd.DataFrame([
        {"entity_id": "S3-9001", "business_name": "Dubois Boulangerie SARL", "business_address": "14 Bd St Germain, Paris 75005", "country": "France"},
        {"entity_id": "S3-9002", "business_name": "Bangalore IT Consulting Pvt Ltd", "business_address": "100 Outer Ring Road, Bangalore 560037", "country": "India"},
    ])

    train_s1.to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    train_s2.to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    train_s3.to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    test_s1.to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
    test_s2.to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
    test_s3.to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)

def test_full_pipeline_synthetic():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = os.path.join(tmp_dir, "dataset")
        output_dir = os.path.join(tmp_dir, "output")
        create_synthetic_dataset(data_dir)

        # Run pipeline
        run_pipeline(data_dir=data_dir, output_dir=output_dir, mode="full")

        # Verify output files exist
        match_tsv = os.path.join(output_dir, "matching_results.tsv")
        cand_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

        assert os.path.exists(match_tsv), "matching_results.tsv missing"
        assert os.path.exists(cand_tsv), "candidate_pairs.tsv missing"

        # Check validator
        val_pass, val_msg = run_submission_validator(
            matching_tsv_path=match_tsv,
            candidate_tsv_path=cand_tsv,
            test_dir=os.path.join(data_dir, "test")
        )

        assert val_pass, f"Validator failed: {val_msg}"

        # Read matches and verify French zero-shot resolution
        df_match = pd.read_csv(match_tsv, sep="\t", dtype=str, keep_default_na=False)
        match_dict = dict(zip(df_match["source1_entity_id"], df_match["matched_entity_ids"]))

        # Check that French entity S1-9001 matched Dubois in S2 and S3
        french_matches = match_dict.get("S1-9001", "").split(",")
        assert "S2-9001" in french_matches or "S3-9001" in french_matches, f"French entity should match candidates, got {french_matches}"

if __name__ == "__main__":
    test_full_pipeline_synthetic()
    print("Full pipeline integration test PASSED!")
