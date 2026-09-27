# Amazon ML Challenge 2026 — Business Entity Resolution

## Problem

Given three sources of business entity records (Source 1, Source 2, Source 3), identify all Source 2 and Source 3 records that correspond to the same real-world entity as each Source 1 record.

- Source 1 is the reference/master source
- Each Source 1 entity may have zero, one, or multiple matches
- Entities span US, India, and France
- Business names and addresses are noisy, inconsistent, and abbreviated

**Metric**: Macro-averaged entity-level **F0.5** (precision-heavy)

---

## Architecture

```
DATA
  ↓
DATA AUDIT (distribution, nulls, match stats)
  ↓
MULTI-VIEW NORMALIZATION (11 name views, 14+ address views)
  ↓
MULTI-CHANNEL BLOCKING (11 channels, union)
  ↓
CANDIDATE UNION + BLOCKING RECALL CHECK
  ↓
FEATURE ENGINEERING (~60 pairwise features)
  ↓
LIGHTGBM CLASSIFIER + HARD NEGATIVE MINING
  ↓
THRESHOLD OPTIMIZATION (sweep over val F0.5)
  ↓
ENTITY-LEVEL DECISION (singleton gate + margin logic)
  ↓
OUTPUT GENERATION + INTERNAL VALIDATION
  ↓
OFFICIAL VALIDATOR
  ↓
SUBMISSION VERSIONING
```

---

## Installation

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

**Required packages:**
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `lightgbm` — main classifier (Apache 2.0 ✓)
- `rapidfuzz` — fast string similarity (MIT ✓)
- `pyyaml`, `tqdm`

---

## Data Setup

Place files in the following structure:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv

utils/
└── validate_submission.py
```

---

## Configuration

All parameters live in `configs/default.yaml`. Key settings:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `blocking` | `tfidf_top_k` | 40 | TF-IDF candidates per S1 entity |
| `blocking` | `rare_token_idf_threshold` | 3.0 | IDF threshold for "rare" tokens |
| `training` | `hard_negative_iterations` | 2 | Hard negative mining rounds |
| `training` | `random_negative_ratio` | 10 | Negatives per positive (initial) |
| `decision` | `default_threshold` | 0.75 | Match probability threshold |
| `decision` | `min_entity_score` | 0.40 | Entity-level gate |
| `model` | `name` | lightgbm | Model backend |
| `random_seed` | — | 42 | Global reproducibility seed |

---

## Running the Full Pipeline

```bash
# From project root (Amazon_Ml_challenge/)
cd code/business_entity_resolution

# Full pipeline (train + infer)
python -m src.main --config configs/default.yaml --phase all

# Train only (saves model to models/)
python -m src.main --config configs/default.yaml --phase train

# Inference only (loads saved model)
python -m src.main --config configs/default.yaml --phase infer
```

---

## Preprocessing

**Business Name** — 11 normalized views:
1. `raw` — original stripped
2. `unicode_norm` — NFKD Unicode normalized
3. `lowercase` — lowercased
4. `punct_norm` — punctuation normalized (& → and)
5. `no_diacritics` — diacritics removed (é→e, ã→a)
6. `cleaned` — full pipeline (all above combined)
7. `alphanumeric` — letters/digits only
8. `tokens` — space-joined word tokens
9. `sorted_tokens` — alphabetically sorted tokens
10. `stripped` — legal suffixes removed (Corp, LLC, Ltd, etc.)
11. `stripped_sorted` — stripped + sorted

**Business Address** — 14+ normalized views plus structural extraction:
- Cleaned/expanded/sorted/alphanumeric text
- `numeric_tokens` — all digit sequences
- `first_numeric` — house/building number candidate
- `postal_candidates` — postal/zip code patterns
- Token count, digit count, char count

**Country**: lowercase, diacritics-free, open-set (no one-hot encoding, France-safe)

---

## Candidate Generation (Blocking)

11 independent blocking channels — candidates are the **UNION**:

| Channel | Strategy |
|---------|----------|
| A | Exact cleaned name |
| B | Exact suffix-stripped name |
| C | Exact sorted-token name |
| D | Name prefix (4, 5, 6 chars) |
| E | Country + 3-char name prefix |
| F | Postal code match |
| G | House number + name prefix |
| H | Shared rare name/address tokens (IDF-filtered) |
| I | TF-IDF cosine — word-level name (top-40) |
| J | TF-IDF cosine — char 3-gram name (top-40) |
| K | TF-IDF cosine — word-level address (top-40) |

**Blocking recall** is computed and logged. Target: ≥ 0.95.

---

## Feature Engineering

~60 pairwise features per candidate pair:

**Name (22 features):**
- Exact match (cleaned, stripped, sorted, stripped-sorted)
- Levenshtein similarity (cleaned, stripped)
- Jaro-Winkler similarity
- Token sort ratio, token set ratio (rapidfuzz)
- Token Jaccard, token overlap, token containment (both directions)
- Stripped token Jaccard
- IDF-weighted Jaccard, rare-token overlap
- TF-IDF word cosine, char n-gram cosine
- Length ratio, length difference, prefix ratio

**Address (14 features):**
- Exact match
- Token Jaccard, overlap, containment
- Numeric token Jaccard, overlap
- House number match (first numeric)
- Postal code match and Jaccard
- IDF-weighted address Jaccard
- TF-IDF word cosine
- Length ratio

**Country (4):** exact match, S1/S2 missing, both missing

**Combined (9):** name×address, name+address, max/min, high-name-low-address, both-high, country×name, country×address

**Missingness (6):** per-field missing indicators

**Source (2):** is_s2_pair, is_s3_pair

**Blocking (1):** channel count

---

## Model Training

1. **Baseline**: LightGBM with 10:1 negative ratio (random negatives from candidates)
2. **Cross-validation**: Entity-level GroupKFold (5 folds, S1 entity as group key)
3. **Threshold optimization**: Sweep [0.50 → 0.98], maximize macro F0.5 on validation
4. **Hard negative mining** (2 iterations):
   - Score all non-match candidates
   - Take top-scoring false positives as hard negatives
   - Retrain on augmented data
5. **Final threshold re-optimization** after last retraining

**Grouped split**: `GroupKFold` with `source1_entity_id` as group — no data leakage.

---

## Validation

```python
from src.evaluation import evaluate, print_evaluation

result = evaluate(predictions, gt_dict, s1_ids)
# Returns: macro_f05, macro_precision, macro_recall,
#          singleton_accuracy, multi_match_avg_f05,
#          source2/source3 breakdown, per_entity list
```

---

## Decision Policy

For each S1 entity:
1. **Entity-level gate**: if `max_score < min_entity_score (0.40)` → empty prediction
2. **Margin check**: if `top - second < 0.15` and `top < 0.95` → raise threshold by +0.10
3. **Per-candidate filter**: accept if `score ≥ source-specific threshold`

Source-specific thresholds (S2 vs S3) are optimized separately on validation data.

---

## Output Files

```
output/
├── matching_results.tsv   — source1_entity_id <TAB> matched_entity_ids
└── candidate_pairs.tsv    — source1_entity_id <TAB> candidate_entity_ids
```

- Every S1 entity appears exactly once
- `matched_entity_ids` is comma-separated or empty (singleton)
- Every match must also appear in candidates

---

## Running the Official Validator

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Submission Versioning

Each run automatically archives to `submissions/v_YYYYMMDD_HHMMSS/`:
- `matching_results.tsv`
- `candidate_pairs.tsv`
- `metadata.json` (config, metrics, timestamp, description)

Experiment metrics are appended to `experiments/results.csv`.

---

## Reproducibility

All random seeds controlled via `random_seed: 42` in config. Pipeline is deterministic given the same data and config.

---

## Project Structure

```
code/business_entity_resolution/
├── src/
│   ├── data.py            Data loading & audit
│   ├── normalization.py   Multi-view text normalization
│   ├── blocking.py        Multi-channel candidate generation
│   ├── features.py        Pairwise feature engineering
│   ├── training.py        LightGBM training + hard negatives
│   ├── evaluation.py      Entity-level F0.5 evaluation
│   ├── inference.py       Test inference pipeline
│   ├── decision.py        Entity-level decision policy
│   ├── submission.py      Output generation & validation
│   └── main.py            Master orchestrator
├── configs/
│   └── default.yaml       Central configuration
├── experiments/
│   └── results.csv        Experiment log (auto-generated)
├── README.md
└── requirements.txt

dataset/
├── train/
└── test/

output/
├── matching_results.tsv
└── candidate_pairs.tsv

submissions/           Versioned submission archives
models/                Saved models and engines
logs/                  Pipeline logs
utils/
└── validate_submission.py  (provided by competition)
```
