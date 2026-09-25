# LinkSure — Business Entity Resolution
### Amazon ML Challenge: Business Entity Resolution

LinkSure is an end-to-end, reproducible, CPU-first Business Entity Resolution pipeline engineered to link noisy, fragmented business records across 3 independent data sources to a deduplicated reference source (Source 1). It is tuned specifically to maximize **macro-averaged $F_{0.5}$** and generalizes zero-shot to an unseen country (France).

---

## 1. Quick Start & Reproduction

### Prerequisites
- Python 3.10 or 3.11 (64-bit)
- Standard multi-core CPU (8 GB+ RAM recommended)
- Zero GPU required

### Installation
From this directory (`code/business_entity_resolution/`):
```bash
python -m pip install -r requirements.txt
```

### Reproduce Final Outputs in One Command
To regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from raw data:
```bash
python src/run.py --data-dir ../../dataset --output-dir ../../output
```

From repository root:
```bash
python -m code.business_entity_resolution.src.run --data-dir dataset --output-dir output
```

### Validate Submission Files
Run the official validator from the repository root:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 2. Pipeline Architecture

LinkSure consists of five core stages:

1. **Preprocessing & Normalization Engine (`src/normalize/`)**:
   - Unicode NFKD decomposition and diacritic stripping ($\text{é} \rightarrow \text{e}$).
   - Country-specific legal suffix extraction and core name stripping (US: `Inc`, `LLC`, `Corp`; India: `Pvt Ltd`, `LLP`; France: `SARL`, `SAS`, `EURL`).
   - Street and locality abbreviation standardization (`bd` $\rightarrow$ `boulevard`, `opp` $\rightarrow$ `opposite`).
   - Accurate 5/6 digit postal and PIN code extraction and landmark isolation.
2. **Multi-Retriever Blocking (`src/blocking/`)**:
   - Union of 5 complementary candidate retrievers:
     - Core name character 3–5-gram TF-IDF
     - Full name + address character 3–5-gram TF-IDF
     - Address word token TF-IDF with IDF weighting
     - Exact postal code / PIN prefix inverted index
     - Acronym and DBA key matching
   - Evaluates $>99.8\%$ space reduction while maintaining $\ge 97\%$ recall ceiling.
   - Outputs `output/candidate_pairs.tsv`.
3. **Feature Engineering Engine (`src/features/`)**:
   - 41 relational, country-agnostic similarity features per pair:
     - RapidFuzz Levenshtein, Jaro-Winkler, token sort/set ratios on clean and core names
     - Token rarity and information content (shared token IDF sum, max shared IDF, max unshared IDF)
     - Tri-state postal code agreement indicators and prefix matching
     - Building number exact match / conflict flags
     - Contextual rank, score gaps, and reverse rank features
4. **Calibrated Matcher (`src/model/`)**:
   - LightGBM binary classifier trained on hard-negative blocked candidates.
   - 5-fold GroupKFold cross-validation grouped strictly on Source 1 entities.
   - Out-of-fold Isotonic Regression calibration converting raw logits into reliable posterior probabilities $P(\text{match})$.
5. **Metric-Aware Decision Layer (`src/decide/`)**:
   - **One-to-One Consistency Resolution:** Enforces that each partner record is claimed exclusively by the reference entity with highest confidence.
   - **Expected-F0.5 Subset Selection:** Directly maximizes per-entity expected $F_{0.5}$ over candidates $k \in \{0, 1, \dots, K\}$, treating singletons as a first-class optimization target.
   - Outputs `output/matching_results.tsv`.

---

## 3. Hardware Requirements & Runtimes

| Stage | Hardware Used | Expected Runtime (10k entities) | Memory Footprint |
|---|---|---|---|
| Normalization | Multi-core CPU | ~15 seconds | < 300 MB |
| Multi-Retriever Blocking | Multi-core CPU | ~45 seconds | < 500 MB (chunked sparse top-k) |
| Feature Engineering | Multi-core CPU | ~1.5 minutes | < 600 MB |
| LightGBM CV & Calibration | Multi-core CPU | ~30 seconds | < 500 MB |
| Expected-F0.5 Selection | Multi-core CPU | ~10 seconds | < 200 MB |
| **Total End-to-End** | **Standard Laptop** | **~3 - 4 minutes** | **< 1 GB RAM** |

---

## 4. Models & Licenses Compliance

All models and third-party libraries strictly adhere to the competition rules (MIT / Apache-2.0 / BSD, $\le 8\text{B}$ parameters, no external API lookups):

| Component / Library | Model / Version | License | Parameter Count | Compliance Note |
|---|---|---|---|---|
| **LightGBM** | LightGBM 4.7.0 GBDT | MIT License | ~3,000 leaf params | Full local CPU execution |
| **scikit-learn** | scikit-learn 1.7.2 | BSD-3-Clause | — | TF-IDF & Isotonic Regression |
| **RapidFuzz** | RapidFuzz 3.14.5 | MIT License | — | C++ string distance engine |
| **pandas & numpy** | pandas 2.3.3, numpy 2.2.6 | BSD-3-Clause | — | High-performance tabular I/O |
| **pyarrow** | pyarrow 25.0.1 | Apache-2.0 | — | Parquet caching |

**Strict Fair-Play Guarantee:** Zero external APIs, commercial entity resolution tools, geocoders, or web augmentation datasets were used.
