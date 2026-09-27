"""
evaluation.py — Entity-level evaluation metrics for the competition.

The competition metric is macro-averaged entity-level F0.5.
F_beta weighs precision beta^2 times more than recall.
F0.5: beta=0.5 → precision is 4x more important than recall.

This module implements:
- f05_score(precision, recall)
- entity_metrics_single(predicted, true) → (p, r, f0.5)
- evaluate(predictions, ground_truth, s1_ids) → full metrics dict
- blocking_recall evaluation
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BETA = 0.5
BETA2 = BETA ** 2


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    """
    Compute F0.5 (precision-heavy F-score).

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    denom = BETA2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + BETA2) * precision * recall / denom


def entity_metrics_single(
    predicted: Set[str], true: Set[str]
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F0.5 for a single S1 entity.

    Args:
        predicted: Set of predicted match IDs.
        true: Set of true match IDs.

    Returns:
        (precision, recall, f05)
    """
    if not predicted and not true:
        # True singleton, correctly identified → perfect score
        return 1.0, 1.0, 1.0

    if not predicted:
        # No prediction made but there are true matches → all missed
        return 0.0, 0.0, 0.0

    if not true:
        # Predicted something but there are no true matches → false merge (score 0.0)
        return 0.0, 0.0, 0.0

    tp = len(predicted & true)
    p = tp / len(predicted)
    r = tp / len(true)
    return p, r, f05_score(p, r)


def evaluate(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Compute macro-averaged entity-level evaluation metrics.

    Args:
        predictions: Dict[s1_id -> list of predicted match IDs]
        ground_truth: Dict[s1_id -> list of true match IDs]
        s1_ids: If provided, restrict evaluation to these S1 IDs.
                Otherwise use keys from ground_truth.

    Returns:
        Dict with keys:
        - macro_f05
        - macro_precision
        - macro_recall
        - singleton_precision (fraction of true singletons correctly left empty)
        - singleton_false_merge (fraction of true singletons with any prediction)
        - multi_match_f05 (average F0.5 over S1 entities with ≥1 true match)
        - per_entity (list of (s1_id, p, r, f05) for all entities)
        - source_breakdown (S2 vs S3 sub-metrics)
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    all_p, all_r, all_f = [], [], []
    singleton_correct = 0
    singleton_false_merge = 0
    singleton_total = 0
    multi_match_f = []
    per_entity = []

    # Source-specific tracking
    s2_tp, s2_fp, s2_fn = 0, 0, 0
    s3_tp, s3_fp, s3_fn = 0, 0, 0

    for s1_id in s1_ids:
        true_list = ground_truth.get(s1_id, [])
        pred_list = predictions.get(s1_id, [])

        true_set = set(true_list)
        pred_set = set(pred_list)

        p, r, f = entity_metrics_single(pred_set, true_set)
        all_p.append(p)
        all_r.append(r)
        all_f.append(f)
        per_entity.append((s1_id, p, r, f))

        # Singleton analysis
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            else:
                singleton_false_merge += 1

        # Multi-match analysis
        if true_set:
            multi_match_f.append(f)

        # Source-specific
        for mid in true_set:
            if mid in pred_set:
                if mid.startswith("S2-"):
                    s2_tp += 1
                else:
                    s3_tp += 1
            else:
                if mid.startswith("S2-"):
                    s2_fn += 1
                else:
                    s3_fn += 1
        for mid in pred_set:
            if mid not in true_set:
                if mid.startswith("S2-"):
                    s2_fp += 1
                else:
                    s3_fp += 1

    macro_p = float(np.mean(all_p))
    macro_r = float(np.mean(all_r))
    macro_f05 = float(np.mean(all_f))

    def _safe_div(a, b):
        return a / b if b else 0.0

    s2_p = _safe_div(s2_tp, s2_tp + s2_fp)
    s2_r = _safe_div(s2_tp, s2_tp + s2_fn)
    s3_p = _safe_div(s3_tp, s3_tp + s3_fp)
    s3_r = _safe_div(s3_tp, s3_tp + s3_fn)

    result = {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "n_entities": len(s1_ids),
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": _safe_div(singleton_correct, singleton_total),
        "singleton_false_merge_rate": _safe_div(singleton_false_merge, singleton_total),
        "multi_match_avg_f05": float(np.mean(multi_match_f)) if multi_match_f else 0.0,
        "source2": {
            "precision": s2_p,
            "recall": s2_r,
            "f05": f05_score(s2_p, s2_r),
            "tp": s2_tp, "fp": s2_fp, "fn": s2_fn,
        },
        "source3": {
            "precision": s3_p,
            "recall": s3_r,
            "f05": f05_score(s3_p, s3_r),
            "tp": s3_tp, "fp": s3_fp, "fn": s3_fn,
        },
        "per_entity": per_entity,
    }

    return result


def print_evaluation(result: Dict, label: str = "Validation"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {label}")
    print(f"{'='*60}")
    print(f"  Macro F0.5  : {result['macro_f05']:.4f}")
    print(f"  Precision   : {result['macro_precision']:.4f}")
    print(f"  Recall      : {result['macro_recall']:.4f}")
    print(f"  N entities  : {result['n_entities']}")
    print(f"  Singletons  : {result['singleton_total']} "
          f"  Correct: {result['singleton_correct']} "
          f"  Accuracy: {result['singleton_accuracy']:.3f}")
    print(f"  Singleton false-merge rate: {result['singleton_false_merge_rate']:.3f}")
    print(f"  Multi-match avg F0.5: {result['multi_match_avg_f05']:.4f}")
    s2 = result["source2"]
    s3 = result["source3"]
    print(f"  Source 2  — P={s2['precision']:.4f} R={s2['recall']:.4f} F05={s2['f05']:.4f}")
    print(f"  Source 3  — P={s3['precision']:.4f} R={s3['recall']:.4f} F05={s3['f05']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Blocking recall evaluation
# ---------------------------------------------------------------------------

def evaluate_blocking(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, List[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Compute blocking recall: fraction of true match pairs that survived blocking.

    A pair that does not survive blocking is unrecoverable by the downstream model.
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    total_true = 0
    in_candidates = 0
    missed_pairs = []

    for s1_id in s1_ids:
        true_set = set(ground_truth.get(s1_id, []))
        cand_set = candidates.get(s1_id, set())
        for mid in true_set:
            total_true += 1
            if mid in cand_set:
                in_candidates += 1
            else:
                missed_pairs.append((s1_id, mid))

    total_cands = sum(len(v) for v in candidates.values())
    avg_per_s1 = total_cands / max(len(s1_ids), 1)
    max_per_s1 = max((len(candidates.get(sid, set())) for sid in s1_ids), default=0)
    recall = in_candidates / max(total_true, 1)

    result = {
        "blocking_recall": recall,
        "total_true_pairs": total_true,
        "pairs_in_candidates": in_candidates,
        "missed_pairs": len(missed_pairs),
        "total_candidates": total_cands,
        "avg_candidates_per_s1": avg_per_s1,
        "max_candidates_per_s1": max_per_s1,
        "missed_sample": missed_pairs[:10],
    }

    print(f"\n  [Blocking] Recall={recall:.4f} ({in_candidates}/{total_true})  "
          f"Avg cands/S1={avg_per_s1:.1f}  Max={max_per_s1}")

    return result


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    top_n: int = 20,
) -> Dict:
    """
    Collect and display the top false positives and false negatives.

    Returns:
        Dict with 'false_positives' and 'false_negatives' lists.
    """
    false_positives = []
    false_negatives = []

    for s1_id, pred_list in predictions.items():
        true_set = set(ground_truth.get(s1_id, []))
        pred_set = set(pred_list)

        for mid in pred_set - true_set:
            false_positives.append({
                "s1_id": s1_id,
                "fp_id": mid,
                "s1_name": s1_lookup.get(s1_id, {}).get("business_name", ""),
                "fp_name": s23_lookup.get(mid, {}).get("business_name", ""),
                "s1_addr": s1_lookup.get(s1_id, {}).get("business_address", ""),
                "fp_addr": s23_lookup.get(mid, {}).get("business_address", ""),
            })

        for mid in true_set - pred_set:
            false_negatives.append({
                "s1_id": s1_id,
                "fn_id": mid,
                "s1_name": s1_lookup.get(s1_id, {}).get("business_name", ""),
                "fn_name": s23_lookup.get(mid, {}).get("business_name", ""),
                "s1_addr": s1_lookup.get(s1_id, {}).get("business_address", ""),
                "fn_addr": s23_lookup.get(mid, {}).get("business_address", ""),
            })

    # Print summaries
    print(f"\n  [Error Analysis] False Positives: {len(false_positives)}, "
          f"False Negatives: {len(false_negatives)}")

    if false_positives:
        print("\n  Top False Positives:")
        for fp in false_positives[:top_n]:
            print(f"    S1={fp['s1_id']} '{fp['s1_name']}' | "
                  f"FP={fp['fp_id']} '{fp['fp_name']}'")

    if false_negatives:
        print("\n  Top False Negatives:")
        for fn in false_negatives[:top_n]:
            print(f"    S1={fn['s1_id']} '{fn['s1_name']}' | "
                  f"FN={fn['fn_id']} '{fn['fn_name']}'")

    return {"false_positives": false_positives, "false_negatives": false_negatives}
