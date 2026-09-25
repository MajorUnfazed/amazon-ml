# LinkSure — Precision-First Business Entity Resolution
### Amazon ML Challenge: Business Entity Resolution Solution

LinkSure is an open-source, reproducible, high-precision Business Entity Resolution system developed for the Amazon ML Challenge.

---

## Repository Structure

```
amazon-ml/
├── dataset/                              # Raw data (train/ and test/ TSVs)
├── output/
│   ├── matching_results.tsv              # Final entity matches (submitted to Portal)
│   └── candidate_pairs.tsv               # Candidate generation pairs for audit
├── utils/
│   └── validate_submission.py            # Official submission validator
├── code/
│   └── business_entity_resolution/       # Runnable, self-contained submission folder
│       ├── src/
│       │   ├── run.py                    # Master one-command entry point
│       │   ├── config.py                 # Hyperparameters & configurations
│       │   ├── data/                     # Schema validation & profiling
│       │   ├── normalize/                # Multi-language normalization rules
│       │   ├── blocking/                 # Multi-retriever union candidate generation
│       │   ├── features/                 # 41 country-agnostic pair features
│       │   ├── model/                    # LightGBM matcher & Isotonic calibration
│       │   ├── decide/                   # 1-to-1 consistency & Expected-F0.5 selection
│       │   ├── pipeline_io/              # TSV export & validator execution
│       │   └── eval/                     # Macro F0.5 evaluation engine
│       ├── tests/                        # Full test suite (14 passing unit & integration tests)
│       ├── README.md                     # Detailed reproduction guide
│       └── requirements.txt              # Pinned dependencies
├── app/                                  # Streamlit Match Explorer (Demo & Error Analysis)
│   └── app.py
├── docs/
│   ├── experiment_log.md                 # Full experiment progression & ablation tracking
│   └── Documentation_template.md         # Filled methodology write-up
├── scripts/
│   ├── make_zip.py                       # Automated submission packaging tool
│   └── make_zip.sh
└── Documentation_template.md             # Root methodology document for zip packaging
```

---

## One-Command Reproduction

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run end-to-end pipeline (data -> blocking -> features -> model -> decisions -> TSV)
python -m code.business_entity_resolution.src.run --data-dir dataset --output-dir output

# 3. Validate output files
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 4. Build final submission zip
python scripts/make_zip.py --team LinkSure
```

---

## Run the Match Explorer UI

To launch the interactive error analysis and match explorer dashboard:
```bash
streamlit run app/app.py
```
