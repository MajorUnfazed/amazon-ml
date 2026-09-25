"""Configuration for LinkSure Business Entity Resolution Pipeline."""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class LinkSureConfig:
    # Directories
    root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "dataset"))
    artifacts_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "artifacts"))
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "output"))

    # Random seed
    random_seed: int = 42

    # Validation
    n_splits: int = 5

    # Blocking configurations
    char_ngram_range: Tuple[int, int] = (3, 5)
    word_ngram_range: Tuple[int, int] = (1, 2)
    top_k_name_tfidf: int = 25
    top_k_name_addr_tfidf: int = 15
    top_k_addr_tfidf: int = 15
    max_candidates_per_s1: int = 50

    # Model configurations (LightGBM)
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 400,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    })

    # Decision Layer
    max_k_expected_f: int = 8
    fallback_threshold: float = 0.50
    enforce_one_to_one: bool = True
    scope_by_country: bool = True

DEFAULT_CONFIG = LinkSureConfig()
