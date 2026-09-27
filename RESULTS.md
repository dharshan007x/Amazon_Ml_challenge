# Amazon ML Challenge 2026 — Business Entity Resolution
# FINAL EXPERIMENTAL RESULTS

> All values below are extracted directly from executed pipeline runs, model artifacts,
> and training/inference logs. Nothing is estimated or fabricated.
> "NOT RUN" = experiment not executed. "NOT AVAILABLE" = metric cannot be derived from current artifacts.

---

## 1. DATASET

| Field | Value |
|---|---|
| **Source 1 records (train)** | 2,206,821 |
| **Source 2 records (train)** | 5,034,616 |
| **Source 3 records (train)** | 5,285,603 |
| **Source 1 records (test)** | 1,732,544 |
| **Source 2 records (test)** | 4,887,273 |
| **Source 3 records (test)** | 5,082,316 |
| **Training S1 entities** | 2,206,821 |
| **Test S1 entities** | 1,732,544 |
| **Countries (train)** | US (1,323,633), India (883,188) |
| **Countries (test)** | US, France, India |
| **Ground-truth rows** | 2,206,821 (one row per S1; each row = comma-separated S2+S3 match IDs) |
| **Total ground-truth matched pairs** | NOT AVAILABLE (requires parsing comma-separated field) |
| **Zero-match S1 entities (train)** | 0 — every training S1 has at least one GT match |

---

## 2. FINAL MODEL

| Field | Value |
|---|---|
| **Model** | LightGBM (LGBMClassifier) |
| **Boosting type** | GBDT |
| **Number of features** | 55 |
| **Best iteration (trees)** | 441 |
| **Max depth** | 7 |
| **Num leaves** | 63 |
| **Learning rate** | 0.05 |
| **Feature fraction** | 0.80 |
| **Bagging fraction** | 0.80 |
| **reg_alpha / reg_lambda** | 0.1 / 0.1 |
| **Random seed** | 42 |
| **Training samples** | 868,094 pairs (from 40,000 S1 entity sample) |
| **Positive samples** | 130,034 |
| **Negative samples** | 738,060 |
| **Positive/negative ratio** | 1 : 5.68 |
| **Hard negatives added** | 2,171 (mining threshold = 0.30) |
| **Training strategy** | Grouped 80/20 split at S1 entity level; negatives from blocking pool; 2-pass hard negative mining |
| **Validation strategy** | S1 entity-level grouped hold-out (~20%); macro F0.5 optimization |
| **Decision threshold (default)** | 0.650 |
| **Decision threshold (S2)** | 0.750 |
| **Decision threshold (S3)** | 0.650 |

---

## 3. CANDIDATE GENERATION

| Field | Value |
|---|---|
| **Blocking strategy** | 8-channel inverted-index: exact cleaned name, exact stripped name, exact sorted-token name, exact alphanumeric name, postal code, house number + first name token, IDF-filtered name tokens (max_bucket=120), IDF-filtered address tokens |
| **Training candidate pairs** | 1,403,900 (for 40,000 S1 sample) |
| **Training avg candidates/S1** | 35.1 |
| **Training candidate recall** | 0.9427 |
| **S1→S2 candidate recall** | NOT AVAILABLE |
| **S1→S3 candidate recall** | NOT AVAILABLE |
| **Test avg candidates/S1 — US** | 31.4 (max: 40) |
| **Test avg candidates/S1 — France** | 28.2 (max: 40) |
| **Test avg candidates/S1 — India** | 30.9 (max: 40) |
| **Candidate cap (top_k)** | 40 per S1 entity |
| **Index keys — US** | 13,854,156 |
| **Index keys — India** | 14,004,508 |
| **Index keys — France** | 4,216,664 |
| **Candidate reduction vs exhaustive (US)** | 663K × 3.82M = 2.53B exhaustive → ~20.8M blocked = **99.2% reduction** |

---

## 4. VALIDATION PERFORMANCE

*Source: `models/pipeline_meta.json` and `logs/pipeline_20260927_015602.log`.*

| Metric | Value |
|---|---|
| **F0.5** | **0.9682** |
| **Precision** | **0.9850** |
| **Recall** | **0.9318** |
| **F1** | 0.9577 |
| **Approx. true positives (TP)** | ~24,234 |
| **Approx. false positives (FP)** | ~369 |
| **Approx. false negatives (FN)** | ~1,773 |
| **Approx. predicted positive pairs** | ~24,603 |
| **Approx. true positive pairs in val** | ~26,007 |
| **S1→S2 F0.5** | NOT AVAILABLE |
| **S1→S3 F0.5** | NOT AVAILABLE |
| **Singleton F0.5** | NOT AVAILABLE |
| **Multi-match F0.5** | NOT AVAILABLE |
| **Zero-match entity performance** | NOT APPLICABLE (no zero-match S1 in training data) |

---

## 5. MODEL COMPARISON

Only LightGBM was trained to final evaluation.

| Model | Features | Precision | Recall | F1 | F0.5 |
|---|---|---|---|---|---|
| **LightGBM (final)** | 55 | **0.9850** | **0.9318** | **0.9577** | **0.9682** |
| Logistic Regression | NOT RUN | — | — | — | — |
| XGBoost | NOT RUN | — | — | — | — |
| CatBoost | NOT RUN | — | — | — | — |

**→ LightGBM is the only model evaluated. Validation F0.5 = 0.9682.**

---

## 6. THRESHOLD EXPERIMENT

*Swept on validation set. Source: `logs/pipeline_20260927_015602.log`.*

| Threshold | Precision | Recall | F1 | F0.5 |
|---|---|---|---|---|
| 0.500 | 0.9821 | 0.9356 | 0.9583 | 0.9670 |
| 0.550 | 0.9830 | 0.9345 | 0.9582 | 0.9673 |
| 0.600 | 0.9842 | 0.9332 | 0.9580 | 0.9678 |
| **0.650** | **0.9850** | **0.9318** | **0.9577** | **0.9682** ← SELECTED |
| 0.700 | 0.9854 | 0.9304 | 0.9571 | 0.9680 |
| 0.750 | 0.9859 | 0.9274 | 0.9558 | 0.9675 |
| 0.800 | 0.9861 | 0.9245 | 0.9543 | 0.9668 |
| 0.820 | 0.9876 | 0.9105 | 0.9474 | 0.9637 |
| 0.840 | 0.9878 | 0.9087 | 0.9466 | 0.9634 |
| 0.860 | 0.9880 | 0.9061 | 0.9452 | 0.9627 |
| 0.880 | 0.9881 | 0.9037 | 0.9440 | 0.9620 |
| 0.900 | 0.9875 | 0.9002 | 0.9419 | 0.9606 |
| 0.920 | 0.9872 | 0.8957 | 0.9393 | 0.9590 |
| 0.940 | 0.9861 | 0.8892 | 0.9351 | 0.9562 |
| 0.960 | 0.9857 | 0.8815 | 0.9306 | 0.9535 |
| 0.980 | 0.9848 | 0.8656 | 0.9214 | 0.9476 |

**Selected threshold: 0.650**
**F0.5 at selected threshold: 0.9682**
**Why selected: Highest validation macro F0.5 across all 16 tested thresholds. 0.650 uniquely maximizes the F0.5 objective; 0.700 is 0.0002 lower.**

S2-specific threshold: **0.750** (optimized independently, F0.5 = 0.9683)
S3-specific threshold: **0.650** (F0.5 = 0.9682)

---

## 7. BLOCKING COMPARISON

Only one configuration was evaluated end-to-end.

| Blocking Strategy | Candidate Count | Candidate Recall | Avg Candidates/S1 |
|---|---|---|---|
| **8-channel inverted index (final)** | 1,403,900 (40K S1 sample) | **0.9427** | **35.1** |
| TF-IDF retrieval channels | NOT RUN end-to-end | NOT AVAILABLE | NOT AVAILABLE |
| Name-exact only | NOT RUN | NOT AVAILABLE | NOT AVAILABLE |

**Final configuration:** 8-channel inverted-index blocking with IDF filtering (max_bucket=120, min_token_len=3).

---

## 8. FEATURE IMPORTANCE

*LightGBM split-based importance from `models/model.pkl`.*

| Rank | Feature | Importance (splits) | Group |
|---|---|---|---|
| 1 | `name_length_ratio` | 1977 | Name similarity |
| 2 | `addr_length_ratio` | 1782 | Address similarity |
| 3 | `name_lev_sim_stripped` | 1704 | Name similarity |
| 4 | `addr_weighted_jaccard` | 1664 | Address similarity |
| 5 | `name_jaro_winkler` | 1512 | Name similarity |
| 6 | `name_token_sort_ratio` | 1451 | Name similarity |
| 7 | `name_length_diff` | 1391 | Name similarity |
| 8 | `addr_token_overlap` | 1139 | Address similarity |
| 9 | `name_weighted_jaccard` | 1059 | Name similarity |
| 10 | `name_lev_sim` | 1021 | Name similarity |
| 11 | `combined_name_plus_addr` | 992 | Cross-field/structural |
| 12 | `name_token_set_ratio` | 983 | Name similarity |
| 13 | `name_prefix_ratio` | 967 | Name similarity |
| 14 | `combined_name_x_addr` | 918 | Cross-field/structural |
| 15 | `addr_token_contain` | 878 | Address similarity |

| Group | Coverage in Top 15 |
|---|---|
| Name similarity | 9 features (ranks 1, 3, 5, 6, 7, 9, 10, 12, 13) |
| Address similarity | 4 features (ranks 2, 4, 8, 15) |
| Cross-field/structural | 2 features (ranks 11, 14) |
| Country | 0 in top 15 |
| Blocking evidence | 0 in top 15 |

---

## 9. ERROR ANALYSIS

| Metric | Value |
|---|---|
| **Approx. false positives** | ~369 |
| **Approx. false negatives** | ~1,773 |
| **Most common FP pattern** | NOT AVAILABLE |
| **Most common FN pattern** | NOT AVAILABLE |
| **Difficult entities** | NOT AVAILABLE |
| **Entities with missing fields** | NOT AVAILABLE |
| **Entities affected by name variation** | NOT AVAILABLE |
| **Entities affected by address variation** | NOT AVAILABLE |
| **Representative FP examples** | NOT AVAILABLE (no per-entity error log) |
| **Representative FN examples** | NOT AVAILABLE (no per-entity error log) |

---

## 10. COUNTRY RESULTS

*Train validation breakdown by country: NOT AVAILABLE (not split per country in logs).*

| Country | S1 (train) | Val Precision | Val Recall | Val F0.5 |
|---|---|---|---|---|
| US | 1,323,633 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| India | 883,188 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| France | 0 (test-only) | N/A | N/A | N/A |

*Test inference match rates (NOT validation F0.5):*

| Country | Test S1 | S2+S3 Pool | Matched | Match Rate |
|---|---|---|---|---|
| US | 663,106 | 3,817,031 | 632,908 | 95.4% |
| France | 259,452 | 1,434,993 | 257,181 | 99.1% |
| India | 809,986 | 4,717,565 | RUNNING | — |

---

## 11. SINGLETON / MULTI-MATCH ANALYSIS

| Metric | Value |
|---|---|
| **S1 with ≥1 GT match (train)** | 2,206,821 (100%) |
| **S1 with zero GT matches (train)** | 0 |
| **Multi-match analysis** | NOT AVAILABLE — GT column `matched_entity_ids` is a comma-separated string; parsing required |
| **Singleton/multi F0.5 breakdown** | NOT AVAILABLE |

---

## 12. FINAL OUTPUT STATISTICS

*India inference still running at time of documentation.*

| Metric | Value |
|---|---|
| **Test S1 entities total** | 1,732,544 |
| **US: with predicted match** | 632,908 |
| **US: with no predicted match** | 30,198 |
| **France: with predicted match** | 257,181 |
| **France: with no predicted match** | 2,271 |
| **India: with predicted match** | RUNNING |
| **Total predicted matches** | NOT AVAILABLE (inference incomplete) |
| **Average predicted matches/S1** | NOT AVAILABLE |
| **Maximum predicted matches for one S1** | NOT AVAILABLE |

---

## 13. FINAL SUBMISSION FILES

| File | Status |
|---|---|
| `output/matching_results.tsv` | NOT GENERATED — India inference running |
| `output/candidate_pairs.tsv` | NOT GENERATED |
| `code/business_entity_resolution/README.md` | ✅ Present |
| `code/business_entity_resolution/requirements.txt` | ✅ Present |
| `Documentation_template.md` | ✅ Present |
| `submission ZIP` | NOT GENERATED |
| **Official validator result** | NOT RUN |
| **Validation errors** | NOT RUN |
| **Duplicate IDs** | NOT RUN |
| **Missing Source 1 IDs** | NOT RUN |
| **Invalid candidate pairs** | NOT RUN |

---

## 14. COPY-PASTE READY VALUES

```
Final model: LightGBM (LGBMClassifier, GBDT)
Number of features: 55
Number of trees (best iteration): 441
Training strategy: Grouped 80/20 split at S1 entity level; negatives from blocking pool; 2-pass hard negative mining (2,171 hard negatives added)
Validation strategy: S1 entity-level grouped hold-out (~20%); macro F0.5 optimization
Validation F0.5: 0.9682
Validation precision: 0.9850
Validation recall: 0.9318
Validation F1: 0.9577
Candidate recall (training): 0.9427
Decision threshold: 0.650 (default), 0.750 (S2), 0.650 (S3)
Training samples: 868,094
Positive samples: 130,034
Negative samples: 738,060 + 2,171 hard negatives
Positive:negative ratio: 1:5.68
Top feature: name_length_ratio (split importance: 1977)
Training avg candidates/S1: 35.1
Training total candidates: 1,403,900 (40K S1 sample)
Test avg candidates/S1 (US): 31.4
Test avg candidates/S1 (France): 28.2
Test avg candidates/S1 (India): 30.9
Train S1: 2,206,821 | Train S2: 5,034,616 | Train S3: 5,285,603
Test S1: 1,732,544  | Test S2: 4,887,273  | Test S3: 5,082,316
Train countries: US (1,323,633), India (883,188)
Test countries: US, France, India
Final test predicted matches: NOT AVAILABLE (inference running)
Official validator status: NOT RUN
```

---

## 15. EXPERIMENT CONCLUSION

- **Best model:** LightGBM (GBDT, 441 trees, 55 features) — the only model evaluated; achieved validation F0.5 = **0.9682**, precision 0.9850, recall 0.9318.
- **Best blocking strategy:** 8-channel inverted-index blocking achieving **candidate recall 0.9427** at 35.1 avg candidates/S1 (training); 99.2% exhaustive pair reduction on US partition.
- **Most important feature group:** Name similarity — 9 of top 15 features are name-based; `name_length_ratio` (1977 splits) is the single most important feature, followed by `name_lev_sim_stripped` and `name_jaro_winkler`.
- **Selected threshold:** **0.650** (default), **0.750** (S2-specific) — selected by F0.5 grid search across 16 thresholds; 0.650 uniquely maximized macro F0.5 = 0.9682 on the held-out validation set.
- **Final validation F0.5: 0.9682** (precision = 0.9850, recall = 0.9318, ~369 FP, ~1,773 FN).
