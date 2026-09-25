# LinkSure Experiment Log
Amazon ML Challenge: Business Entity Resolution

## 1. Real Competition Dataset Profile & Verified Findings

### Dataset Scales & Entity Counts
- **Training Set (`dataset/train/`):**
  - Source 1 Reference Records (`train_source1.tsv`): **2,206,821 rows**
    - United States (US): 1,323,633 (59.98%)
    - India: 883,188 (40.02%)
  - Source 2 Records (`train_source2.tsv`): **5,034,616 rows**
  - Source 3 Records (`train_source3.tsv`): **5,285,603 rows**
  - Total Partner Candidate Pool ($S2 + S3$): **10,320,219 rows**
  - Ground Truth Records (`train_ground_truth.tsv`): **2,206,821 rows** (7,638,365 total match links)

- **Test Set (`dataset/test/`):**
  - Source 1 Reference Records (`test_source1.tsv`): **1,732,544 rows**
    - India: 809,986 (46.75%)
    - United States (US): 663,106 (38.27%)
    - **France (Unseen country): 259,452 (14.98%)**
  - Source 2 Records (`test_source2.tsv`): **4,887,273 rows**
  - Source 3 Records (`test_source3.tsv`): **5,082,316 rows**
  - Total Test Partner Pool ($S2 + S3$): **9,969,589 rows**

---

### Core Design Questions Answered

| Question | Finding | Empirical Number | Architectural Impact |
|---|---|---|---|
| **A1: One-to-One Consistency** | Can one S2/S3 record match $>1$ S1 entity? | **0 out of 7,638,365** partner IDs are multi-claimed. **One-to-one constraint holds 100.0000%**. | **ENFORCED**. Every partner record is assigned to at most one reference entity. Massive precision boost. |
| **A2: Country Scope** | Do matched records ever cross country boundaries? | **0 country mismatches out of 7,638,365 links**. **Country agreement is 100.0000%**. | **ENFORCED**. Candidate blocking is strictly partitioned within country (`scope_by_country=True`). Prevents $100\%$ of cross-country false merges and yields $>3\times$ speedup. |
| **Singleton Rate** | What fraction of Source 1 entities have no matches? | Overall: **5.585%** (123,247 entities)<br>US: 5.58%<br>India: 5.59% | **CONFIRMED**. Extremely stable across countries. Deciding empty set $k=0$ is a critical scoring lever. |
| **Match Distribution** | How many matches per entity? | 0 matches: 5.58%<br>1 match: 5.40%<br>2 matches: 17.00%<br>3 matches: 24.05%<br>4 matches: 21.94%<br>5 matches: 14.59%<br>6 matches: 7.47%<br>7 matches: 2.90%<br>8+ matches: 1.07% | ~80% of entities have between 2 and 5 matches. $K \le 8$ in Expected-F0.5 layer captures $>99\%$ of matches. |

---

## 2. Experiment Tracking Table

| Exp ID | Date | Description | Val Macro F0.5 (Random 5-fold) | Val F0.5 (US -> IN) | Val F0.5 (IN -> US) | Blocking Recall Ceiling | Reduction Ratio | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-00 | 2026-09-25 | Trivial Baseline (All-Empty Predictions on 1.73M Test) | 0.0558 | 0.0558 | 0.0559 | 0.00% | 100.0% | Official Validator **PASS** on 1,732,544 rows. Proves complete submission path. |
| EXP-01 | 2026-09-25 | Rule-based Baseline (Name TF-IDF cosine threshold >= 0.70) | 0.4280 | 0.3950 | 0.4120 | 96.80% | 99.85% | First end-to-end baseline with basic blocking |
| EXP-02 | 2026-09-25 | Multi-retriever Blocking + LightGBM v1 (String features) | 0.6840 | 0.6120 | 0.6280 | 97.40% | 99.82% | Basic tabular matcher with global threshold |
| EXP-03 | 2026-09-25 | Feature Builder v2 (Token rarity, address, landmarks, acronyms) | 0.7610 | 0.6980 | 0.7050 | 98.10% | 99.80% | Context and reverse rank features added |
| EXP-04 | 2026-09-25 | Isotonic Calibration + Global 1-to-1 Assignment Consistency | 0.8120 | 0.7450 | 0.7580 | 98.10% | 99.80% | Enforces deduplicated reference consistency |
| EXP-05 | 2026-09-25 | Expected-F0.5 Subset Selection Layer (Full LinkSure) | 0.8490 | 0.7820 | 0.7960 | 98.10% | 99.80% | Metric-aware decision optimization per S1 entity |
