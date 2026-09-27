"""
inference.py — Full test inference pipeline.

Orchestrates:
1. Normalization of test data
2. Blocking (candidate generation) on test S2+S3
3. Feature computation for all candidate pairs
4. Model scoring
5. Decision policy application
6. Returning final predictions and candidate sets
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .normalization import normalize_dataframe
from .blocking import BlockingEngine
from .features import FeatureEngine
from .decision import DecisionEngine
from .evaluation import evaluate_blocking

logger = logging.getLogger(__name__)


def run_inference(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model,
    feature_engine: FeatureEngine,
    blocking_engine: BlockingEngine,
    decision_engine: DecisionEngine,
    batch_size: int = 5000,
    gt_dict: Optional[Dict[str, List[str]]] = None,  # For blocking recall on val
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """
    Run the full inference pipeline for a given S1/S2/S3 split.

    Args:
        s1_df: Normalized Source 1 DataFrame.
        s2_df: Normalized Source 2 DataFrame.
        s3_df: Normalized Source 3 DataFrame.
        model: Fitted classifier with predict_proba.
        feature_engine: Fitted FeatureEngine.
        blocking_engine: Fitted BlockingEngine.
        decision_engine: DecisionEngine with thresholds set.
        batch_size: Pairs per feature computation batch.
        gt_dict: Ground truth (optional; used for blocking recall reporting only).

    Returns:
        (predictions, candidates, pair_scores)
        - predictions: {s1_id -> [matched_s23_ids]}
        - candidates: {s1_id -> set(candidate_s23_ids)}
        - pair_scores: {s1_id -> {s23_id -> probability}}
    """
    s1_ids = s1_df["entity_id"].tolist()

    # --- Entity lookups (by entity_id -> row dict) ---
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s2_lookup = s2_df.set_index("entity_id").to_dict("index")
    s3_lookup = s3_df.set_index("entity_id").to_dict("index")
    s23_lookup = {**s2_lookup, **s3_lookup}

    # --- Step 1: Generate candidates ---
    logger.info("Inference: generating candidates...")
    candidates = blocking_engine.get_candidates(s1_df)

    # Ensure every S1 ID has an entry
    for s1_id in s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()

    # Optional: report blocking recall on validation set
    if gt_dict is not None:
        evaluate_blocking(candidates, gt_dict, s1_ids)

    # --- Step 2: Build flat pair list ---
    all_pairs: List[Tuple[str, str]] = []
    for s1_id in s1_ids:
        for s23_id in candidates[s1_id]:
            if s23_id in s23_lookup:  # only valid IDs
                all_pairs.append((s1_id, s23_id))

    logger.info("Total candidate pairs to score: %d", len(all_pairs))

    # --- Step 3: Score pairs in batches ---
    pair_scores: Dict[str, Dict[str, float]] = {s1_id: {} for s1_id in s1_ids}

    for start in range(0, len(all_pairs), batch_size):
        batch = all_pairs[start : start + batch_size]
        X_batch, feat_names = feature_engine.compute_batch(batch, s1_lookup, s23_lookup)

        if X_batch.shape[0] == 0:
            continue

        scores = model.predict_proba(X_batch)[:, 1]

        for (s1_id, s23_id), score in zip(batch, scores):
            pair_scores[s1_id][s23_id] = float(score)

        if (start // batch_size) % 10 == 0:
            logger.info("  Scored %d / %d pairs...", start + len(batch), len(all_pairs))

    # --- Step 4: Apply decision policy ---
    logger.info("Applying decision policy...")
    predictions, stats = decision_engine.decide_with_stats(pair_scores, s1_ids)

    # Guarantee every S1 ID has an entry (even if not in pair_scores)
    for s1_id in s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = []

    logger.info("Inference complete. Matches: %d / %d S1 entities",
                stats["n_with_predictions"], stats["n_s1_entities"])

    return predictions, candidates, pair_scores


def score_pairs_only(
    pairs: List[Tuple[str, str]],
    model,
    feature_engine: FeatureEngine,
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    batch_size: int = 5000,
) -> np.ndarray:
    """
    Score a flat list of (s1_id, s23_id) pairs and return probability array.

    Useful for hard negative mining and threshold sweeping.
    """
    all_scores = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        X_batch, _ = feature_engine.compute_batch(batch, s1_lookup, s23_lookup)
        if X_batch.shape[0] == 0:
            all_scores.extend([0.0] * len(batch))
            continue
        scores = model.predict_proba(X_batch)[:, 1]
        all_scores.extend(scores.tolist())

    return np.array(all_scores, dtype=np.float32)
