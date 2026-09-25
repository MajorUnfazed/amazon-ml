"""Submission TSV writer and validator caller for LinkSure."""

import os
import subprocess
import sys
from typing import Dict, Set, List, Optional, Tuple

def write_tsv_submission(
    filepath: str,
    predictions: Dict[str, Set[str]],
    all_s1_ids: List[str],
    id_col_name: str = "source1_entity_id",
    list_col_name: str = "matched_entity_ids"
):
    """
    Writes TSV output strictly adhering to competition formatting rules.
    - One row per Source 1 entity in all_s1_ids
    - Sorted by source1_entity_id
    - Tab-separated, no quotes
    - Comma-separated list with no spaces
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sorted_s1_ids = sorted(all_s1_ids)

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{id_col_name}\t{list_col_name}\n")
        for s1_id in sorted_s1_ids:
            matches = predictions.get(s1_id, set())
            # Deduplicate and sort for determinism
            sorted_matches = sorted(list(set(matches)))
            match_str = ",".join(sorted_matches)
            f.write(f"{s1_id}\t{match_str}\n")

def run_submission_validator(
    matching_tsv_path: str,
    candidate_tsv_path: str,
    test_dir: str,
    validator_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Calls utils/validate_submission.py and captures output.
    Returns (passed, output_message).
    """
    if validator_path is None or not os.path.exists(validator_path):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "..", "utils", "validate_submission.py")),
            os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "utils", "validate_submission.py")),
            os.path.abspath(os.path.join(cur_dir, "..", "..", "utils", "validate_submission.py")),
            os.path.abspath(os.path.join(os.getcwd(), "utils", "validate_submission.py")),
            os.path.abspath(os.path.join(os.getcwd(), "student_resource", "utils", "validate_submission.py"))
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                validator_path = p
                break

    if not os.path.exists(validator_path):
        return False, f"Validator script not found at {validator_path}"

    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_tsv_path,
        "--candidate", candidate_tsv_path,
        "--test-dir", test_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    passed = (result.returncode == 0) and ("PASS" in result.stdout)
    out_msg = result.stdout if result.stdout else result.stderr
    return passed, out_msg
