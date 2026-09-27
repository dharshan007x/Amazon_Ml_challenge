# Business Entity Resolution — Competition Methodology

## 1. Problem Statement

We address cross-source business entity resolution: given a reference set (Source 1) of business records, identify all matching records from Sources 2 and 3. Entities may have zero, one, or multiple cross-source matches. The task is evaluated by macro-averaged entity-level F0.5, which weights precision four times more than recall.

## 2. Candidate Generation (Blocking)

To handle the quadratic comparison space, we implement an 11-channel blocking system. Each channel independently generates candidate pairs for each Source 1 entity, and we take the **union** across channels.

**Deterministic channels (A–H):**
- Exact cleaned/stripped/sorted name matches
- Name prefix blocking (4, 5, 6-character keys)
- Country + name prefix composite keys
- Postal code candidate matching
- House number + name prefix composite keys
- Rare-token matching via inverted index (IDF-filtered)

**Retrieval channels (I–K):**
- Word-level TF-IDF cosine similarity (top-40 per entity)
- Character 3-4-gram TF-IDF cosine similarity (top-40)
- Address word TF-IDF cosine similarity (top-40)

All TF-IDF retrieval uses L2-normalized sparse matrices and batch dot products, avoiding O(N²) loops. Blocking recall is measured on training data; we require ≥0.95 to ensure the downstream classifier can recover true matches.

## 3. Text Normalization

We produce multiple normalized views to handle the diversity of business name representations:

**Business names**: raw, Unicode (NFKD), diacritics-removed, lowercased, punctuation-normalized, alphanumeric-only, tokenized, sorted-token, legal-suffix-stripped, stripped-sorted (11 views).

**Addresses**: cleaned, abbreviation-expanded (rd→road, st→street, etc.), alphanumeric, tokenized, sorted-token, plus structural extractions: all numeric tokens, first numeric (house number candidate), postal code candidates (5-6 digit and alphanumeric patterns), token/digit/char counts.

Country is treated as a raw string (lowercase, diacritics-removed) with no country-specific hardcoding, making the pipeline naturally extensible to unseen countries (e.g., France in the test set).

## 4. Feature Engineering

We compute approximately 60 pairwise features for each candidate pair:

| Feature Group | Count | Key Features |
|--------------|-------|-------------|
| Name similarity | 22 | Levenshtein, Jaro-Winkler, token Jaccard, IDF-weighted Jaccard, TF-IDF cosine (word + char), prefix ratio |
| Address similarity | 14 | Token Jaccard, numeric overlap, house-number match, postal-code match, IDF-weighted Jaccard |
| Country | 4 | Exact match, missing indicators |
| Combined | 9 | name × address, high-low indicators, country interaction terms |
| Missingness | 6 | Per-field absence indicators (missing ≠ disagreement) |
| Source | 2 | is_s2_pair, is_s3_pair |
| Blocking | 1 | Number of channels that generated this pair |

Token IDF weights are computed from the training corpus, giving rare tokens higher weight than common terms ("restaurant", "road", "india").

## 5. Model

**Primary model**: LightGBM (Apache 2.0 license, <1M parameters).

LightGBM is chosen for structured tabular features with heterogeneous scales and missing values. It handles categorical-like features without encoding, supports early stopping, and produces well-calibrated probabilities.

**No external models or large language models are used.** The problem is framed as structured similarity ranking, where tree-based models excel.

## 6. Training Strategy

**Pair construction**: Positive pairs from ground truth; negative pairs sampled from the candidate pool (not from the full Cartesian product, avoiding easy negatives).

**Grouped validation**: We split at the Source 1 entity level using `GroupKFold`, ensuring no S1 entity appears in both train and validation. This prevents data leakage and gives an unbiased estimate of generalization.

**Hard negative mining** (2 iterations):
1. Train baseline on random negatives.
2. Score all non-match candidates with the current model.
3. High-scoring non-matches (false positives) → hard negatives.
4. Retrain on augmented data.

Hard negatives specifically target the failure modes of the current model, dramatically reducing false positives.

## 7. F0.5 Optimization

**Threshold sweep**: We evaluate thresholds from 0.50 to 0.98 in fine steps, selecting the one maximizing macro F0.5 on the validation set. We also optimize source-specific thresholds for S2 and S3 independently if doing so improves overall F0.5.

**Entity-level gate**: If the maximum candidate score for an S1 entity is below 0.40, we output an empty match list. This prevents weak evidence from forcing spurious matches.

**Score margin**: If the gap between the top and second candidate scores is small (< 0.15) and the top score is below 0.95, we raise the effective threshold by 0.10. This handles ambiguous cases where multiple candidates appear similar.

## 8. Singleton Handling

Singleton detection is treated as a first-class objective because:
- False merges (predicting a match when there is none) cost F0.5 heavily
- The entity-level gate ensures entities with weak evidence output empty predictions
- The threshold is optimized to balance singleton precision against match recall

## 9. Validation Framework

We compute entity-level macro F0.5 exactly matching the competition definition:

```
F_0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
```

Evaluated per S1 entity (including singletons), then macro-averaged. We report separately: singleton accuracy, multi-match F0.5, S2/S3 breakdown, false positive count, false negative count.

Error analysis categorizes failures into: blocking failures (missed candidate), threshold failures, name collision (different businesses with similar names), address collision, typo/abbreviation issues, and missing-field issues.

## 10. Conclusion

Our system combines high-recall multi-channel blocking with precision-optimized LightGBM classification. The key advantages are:
- **Coverage**: 11 blocking channels ensuring high recall even for noisy/abbreviated records
- **Precision**: Hard negative mining and conservative thresholding targeting F0.5
- **Robustness**: No country-specific hardcoding; normalization handles diacritics, abbreviations, legal suffixes across any language
- **Transparency**: All components are interpretable; feature importance from LightGBM directly shows which signals matter most
