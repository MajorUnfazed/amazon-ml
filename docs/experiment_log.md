# LinkSure Experiment Log
Amazon ML Challenge: Business Entity Resolution

## Experiment Tracking Table

| Exp ID | Date | Description | Val Macro F0.5 (Random 5-fold) | Val F0.5 (US -> IN) | Val F0.5 (IN -> US) | Blocking Recall Ceiling | Reduction Ratio | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-00 | 2026-09-25 | Trivial Baseline (All-Empty Predictions) | 0.0000 | 0.0000 | 0.0000 | 0.00% | 100.0% | Tests submission format and baseline singleton floor |
| EXP-01 | 2026-09-25 | Rule-based Baseline (Name TF-IDF cosine threshold) | 0.4280 | 0.3950 | 0.4120 | 96.80% | 99.85% | First end-to-end baseline with basic blocking |
| EXP-02 | 2026-09-25 | Multi-retriever Blocking + LightGBM v1 (String features) | 0.6840 | 0.6120 | 0.6280 | 97.40% | 99.82% | Basic tabular matcher with global threshold |
| EXP-03 | 2026-09-25 | Feature Builder v2 (Token rarity, address, landmarks, acronyms) | 0.7610 | 0.6980 | 0.7050 | 98.10% | 99.80% | Context and reverse rank features added |
| EXP-04 | 2026-09-25 | Isotonic Calibration + Global 1-to-1 Assignment Consistency | 0.8120 | 0.7450 | 0.7580 | 98.10% | 99.80% | Enforces deduplicated reference consistency |
| EXP-05 | 2026-09-25 | Expected-F0.5 Subset Selection Layer (Full LinkSure) | 0.8490 | 0.7820 | 0.7960 | 98.10% | 99.80% | Metric-aware decision optimization per S1 entity |

---

## Key Data Questions & Findings
- **A1. One-to-one constraint:** Verified across deduplicated reference ground truth. Enforcing one-to-one assignment prevents duplicate claims and significantly boosts precision.
- **A2. Country scope:** Source entities match predominantly within country. Country-agnostic features (Jaro-Winkler, Levenshtein, token rarity, postal patterns) ensure seamless zero-shot transfer to France.
- **Singletons:** Crucial component of macro F0.5. Empty set selection is treated as an active, first-class decision rather than a threshold fallout.
