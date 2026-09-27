"""
training.py — Model training pipeline with hard negative mining.

Steps:
1. Build initial positive + random negative training pairs from candidates
2. Train a LightGBM baseline classifier
3. Hard negative mining: find high-confidence false positives → add to negatives
4. Retrain on augmented negatives
5. Cross-validate (entity-level GroupKFold) to estimate F0.5
6. Threshold optimization (sweep over validation probabilities)
"""

import logging
import os
import pickle
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logging.getLogger(__name__).warning("lightgbm not installed.")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from .evaluation import evaluate, print_evaluation, f05_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    gt_dict: Dict[str, List[str]],
    candidates: Dict[str, Set[str]],
    negative_ratio: int = 10,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Build positive and negative training pairs from candidate set.

    Positive pairs: all ground-truth S1→S23 matches.
    Negative pairs: candidate pairs NOT in ground truth, sampled to
                    `negative_ratio` negatives per positive.

    Args:
        s1_df: Normalized Source 1 DataFrame.
        s23_df: Normalized combined S2+S3 DataFrame.
        gt_dict: Ground truth {s1_id -> [match_ids]}.
        candidates: Blocking candidates {s1_id -> set(s23_ids)}.
        negative_ratio: Negatives per positive for initial sampling.
        random_seed: For reproducibility.

    Returns:
        (positive_pairs, negative_pairs)
    """
    rng = random.Random(random_seed)

    s1_ids = set(s1_df["entity_id"].tolist())
    s23_ids = set(s23_df["entity_id"].tolist())

    positive_pairs: List[Tuple[str, str]] = []
    negative_pairs: List[Tuple[str, str]] = []

    for s1_id in s1_ids:
        true_matches = set(gt_dict.get(s1_id, []))
        cand_set = candidates.get(s1_id, set()) & s23_ids

        # Positives (only those in candidates)
        for mid in true_matches:
            if mid in cand_set:
                positive_pairs.append((s1_id, mid))

        # Negatives (candidate - true_matches)
        neg_pool = list(cand_set - true_matches)
        n_neg = min(len(neg_pool), max(1, len(true_matches) * negative_ratio) if true_matches else min(5, len(neg_pool)))
        if neg_pool:
            sampled_negs = rng.sample(neg_pool, min(n_neg, len(neg_pool)))
            for mid in sampled_negs:
                negative_pairs.append((s1_id, mid))

    logger.info(
        "Training pairs — positives: %d, negatives: %d (ratio: %.1f)",
        len(positive_pairs), len(negative_pairs),
        len(negative_pairs) / max(len(positive_pairs), 1),
    )
    return positive_pairs, negative_pairs


def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    labels: List[int],
    feature_engine,
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    channel_counts: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute feature matrix X and label vector y for a list of pairs.

    Returns:
        (X, y, feature_names)
    """
    if not pairs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int32), []

    X, feat_names = feature_engine.compute_batch(pairs, s1_lookup, s23_lookup, channel_counts)
    y = np.array(labels, dtype=np.int32)
    return X, y, feat_names


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    model,
    pos_pairs: List[Tuple[str, str]],
    all_candidates: Dict[str, Set[str]],
    gt_dict: Dict[str, List[str]],
    feature_engine,
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    hard_negative_ratio: int = 4,
    score_threshold: float = 0.3,
    channel_counts: Optional[Dict] = None,
    random_seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Mine hard negatives: non-match candidate pairs that the model scores highly.

    These are false positives from the current model — they are the most
    informative negatives for the next training iteration.

    Args:
        model: Trained classifier with predict_proba method.
        pos_pairs: Current positive pairs (to know true matches).
        all_candidates: Blocking candidates {s1_id -> set(s23_ids)}.
        gt_dict: Ground truth.
        feature_engine: Fitted FeatureEngine.
        s1_lookup, s23_lookup: Entity lookups.
        hard_negative_ratio: Max hard negatives per positive.
        score_threshold: Minimum score to be considered a hard negative.

    Returns:
        List of hard negative pairs.
    """
    rng = random.Random(random_seed)

    # Build set of true pairs for quick lookup
    pos_set = set(pos_pairs)

    hard_negs = []
    n_pos = len(pos_pairs)
    max_hard = n_pos * hard_negative_ratio

    # Collect all non-match candidates and score them
    all_neg_candidates = []
    for s1_id, cand_set in all_candidates.items():
        true_matches = set(gt_dict.get(s1_id, []))
        for s23_id in cand_set:
            if (s1_id, s23_id) not in pos_set and s23_id not in true_matches:
                all_neg_candidates.append((s1_id, s23_id))

    if not all_neg_candidates:
        return []

    # Score all negative candidates
    rng.shuffle(all_neg_candidates)
    batch_size = 5000
    all_scores = []

    for start in range(0, len(all_neg_candidates), batch_size):
        batch = all_neg_candidates[start : start + batch_size]
        X_batch, _ = feature_engine.compute_batch(batch, s1_lookup, s23_lookup, channel_counts)
        if X_batch.shape[0] == 0:
            continue
        scores = model.predict_proba(X_batch)[:, 1]
        all_scores.extend(zip(scores, batch))

    # Sort by score descending, take top hard negatives above threshold
    all_scores.sort(key=lambda x: x[0], reverse=True)

    for score, pair in all_scores:
        if score < score_threshold:
            break
        if len(hard_negs) >= max_hard:
            break
        hard_negs.append(pair)

    logger.info("Hard negative mining: found %d hard negatives (threshold=%.2f).",
                len(hard_negs), score_threshold)
    return hard_negs


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    early_stopping_rounds: int = 50,
) -> "lgb.LGBMClassifier":
    """
    Train a LightGBM classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data for early stopping (optional).
        params: LightGBM hyperparameters.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Fitted LGBMClassifier.
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm is required. Install: pip install lightgbm")

    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    else:
        # No early stopping — use all estimators
        model.fit(X_train, y_train)

    logger.info("LightGBM trained. Best iteration: %s", getattr(model, "best_iteration_", "N/A"))
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    model_config: dict,
) -> object:
    """
    Train the configured model.

    Dispatches to the correct backend based on model_config['name'].
    """
    name = model_config.get("name", "lightgbm").lower()
    params = model_config.get("params", {})
    early_stopping = model_config.get("early_stopping_rounds", 50)

    if name == "lightgbm":
        return train_lightgbm(X_train, y_train, X_val, y_val, params, early_stopping)
    else:
        raise ValueError(f"Unsupported model: {name}. Supported: lightgbm")


# ---------------------------------------------------------------------------
# Cross-validation (entity-level)
# ---------------------------------------------------------------------------

def cross_validate_entity_level(
    pairs: List[Tuple[str, str]],
    labels: List[int],
    feature_engine,
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    gt_dict: Dict[str, List[str]],
    candidates: Dict[str, Set[str]],
    model_config: dict,
    n_folds: int = 5,
    threshold: float = 0.75,
    random_seed: int = 42,
) -> Dict:
    """
    Entity-level cross-validation.

    Groups pairs by S1 entity ID so the same S1 entity never appears in
    both train and validation.

    Returns:
        Dict with fold results and mean metrics.
    """
    s1_ids_per_pair = [p[0] for p in pairs]
    unique_s1_ids = list(set(s1_ids_per_pair))

    splitter = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros((len(unique_s1_ids), 1))
    groups = np.arange(len(unique_s1_ids))

    # Map s1_id to index
    s1_to_idx = {sid: i for i, sid in enumerate(unique_s1_ids)}
    pair_groups = np.array([s1_to_idx[p[0]] for p in pairs])

    fold_results = []
    pair_arr = np.array(pairs, dtype=object)
    label_arr = np.array(labels)

    logger.info("Cross-validating with %d folds (entity-level split)...", n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(dummy_X, groups=groups)
    ):
        # Map fold indices to pair indices
        train_s1_set = {unique_s1_ids[i] for i in train_idx}
        val_s1_set = {unique_s1_ids[i] for i in val_idx}

        pair_train_mask = np.array([p[0] in train_s1_set for p in pairs])
        pair_val_mask = ~pair_train_mask

        train_pairs = [pairs[i] for i in range(len(pairs)) if pair_train_mask[i]]
        val_pairs = [pairs[i] for i in range(len(pairs)) if not pair_train_mask[i]]
        train_labels = label_arr[pair_train_mask].tolist()
        val_labels = label_arr[pair_val_mask].tolist()

        if not train_pairs or not val_pairs:
            continue

        X_train, y_train, _ = build_feature_matrix(
            train_pairs, train_labels, feature_engine, s1_lookup, s23_lookup
        )
        X_val, y_val, _ = build_feature_matrix(
            val_pairs, val_labels, feature_engine, s1_lookup, s23_lookup
        )

        model = train_model(X_train, y_train, X_val, y_val, model_config)

        # Generate predictions for val S1 entities
        val_scores = model.predict_proba(X_val)[:, 1]

        # Reconstruct entity-level predictions
        val_s1_ids = sorted(val_s1_set)
        predictions = {}
        pair_score_map: Dict[str, Dict[str, float]] = {}

        for (s1_id, s23_id), score in zip(val_pairs, val_scores):
            if s1_id not in pair_score_map:
                pair_score_map[s1_id] = {}
            pair_score_map[s1_id][s23_id] = float(score)

        for s1_id in val_s1_ids:
            scores_for_s1 = pair_score_map.get(s1_id, {})
            predictions[s1_id] = [
                mid for mid, sc in scores_for_s1.items() if sc >= threshold
            ]
            # Also add S1 entities that had no candidates at all
            if s1_id not in predictions:
                predictions[s1_id] = []

        # Also add S1 val entities with no pairs (empty predictions)
        for s1_id in val_s1_ids:
            if s1_id not in predictions:
                predictions[s1_id] = []

        # Evaluate
        val_gt = {sid: gt_dict.get(sid, []) for sid in val_s1_ids}
        result = evaluate(predictions, val_gt, val_s1_ids)

        logger.info(
            "Fold %d — F0.5: %.4f  P: %.4f  R: %.4f  Singleton acc: %.3f",
            fold_idx + 1,
            result["macro_f05"],
            result["macro_precision"],
            result["macro_recall"],
            result["singleton_accuracy"],
        )
        fold_results.append(result)

    mean_f05 = float(np.mean([r["macro_f05"] for r in fold_results]))
    mean_p = float(np.mean([r["macro_precision"] for r in fold_results]))
    mean_r = float(np.mean([r["macro_recall"] for r in fold_results]))
    std_f05 = float(np.std([r["macro_f05"] for r in fold_results]))

    logger.info(
        "CV result — F0.5: %.4f ± %.4f  P: %.4f  R: %.4f",
        mean_f05, std_f05, mean_p, mean_r,
    )

    return {
        "mean_f05": mean_f05,
        "std_f05": std_f05,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def optimize_threshold(
    pair_scores: Dict[str, Dict[str, float]],
    gt_dict: Dict[str, List[str]],
    s1_ids: List[str],
    thresholds: Optional[List[float]] = None,
    source_specific: bool = False,
) -> Dict:
    """
    Sweep candidate thresholds and return the one maximizing macro F0.5.

    Args:
        pair_scores: {s1_id -> {s23_id -> score}}
        gt_dict: Ground truth.
        s1_ids: S1 IDs to evaluate on.
        thresholds: List of thresholds to sweep.
        source_specific: If True, also optimize separate thresholds for S2/S3.

    Returns:
        Dict with best_threshold, best_f05, threshold_results.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                      0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

    best_threshold = thresholds[0]
    best_f05 = -1.0
    results = []

    for thr in thresholds:
        predictions = {}
        for s1_id in s1_ids:
            scores_for_s1 = pair_scores.get(s1_id, {})
            predictions[s1_id] = [
                mid for mid, sc in scores_for_s1.items() if sc >= thr
            ]

        metrics = evaluate(predictions, gt_dict, s1_ids)
        f05 = metrics["macro_f05"]
        results.append({"threshold": thr, "f05": f05, "p": metrics["macro_precision"],
                        "r": metrics["macro_recall"]})

        logger.info("  Threshold=%.3f → F0.5=%.4f  P=%.4f  R=%.4f",
                    thr, f05, metrics["macro_precision"], metrics["macro_recall"])

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thr

    logger.info("Best threshold: %.3f (F0.5=%.4f)", best_threshold, best_f05)

    # Source-specific thresholds
    best_s2_thr = best_threshold
    best_s3_thr = best_threshold

    if source_specific:
        for source_prefix, thr_key in [("S2-", "s2"), ("S3-", "s3")]:
            best_src_f05 = -1.0
            best_src_thr = best_threshold

            for thr in thresholds:
                predictions = {}
                for s1_id in s1_ids:
                    scores_for_s1 = pair_scores.get(s1_id, {})
                    preds = []
                    for mid, sc in scores_for_s1.items():
                        src_thr = thr if mid.startswith(source_prefix) else best_threshold
                        if sc >= src_thr:
                            preds.append(mid)
                    predictions[s1_id] = preds

                metrics = evaluate(predictions, gt_dict, s1_ids)
                if metrics["macro_f05"] > best_src_f05:
                    best_src_f05 = metrics["macro_f05"]
                    best_src_thr = thr

            logger.info("  Best %s threshold: %.3f (F0.5=%.4f)", thr_key, best_src_thr, best_src_f05)
            if source_prefix == "S2-":
                best_s2_thr = best_src_thr
            else:
                best_s3_thr = best_src_thr

    return {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "best_s2_threshold": best_s2_thr,
        "best_s3_threshold": best_s3_thr,
        "threshold_results": results,
    }


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------

def save_model(model, path: str):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: str):
    """Load model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
