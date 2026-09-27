"""
decision.py — Entity-level decision logic for final match predictions.

Given candidate pairs with probability scores, this module determines:
- Which candidates exceed the acceptance threshold (per-candidate decision)
- Whether an S1 entity has ANY convincing match (entity-level gating)
- Whether to output an empty match list (singleton detection)
- How to handle multi-match scenarios

Key design for F0.5 (precision-heavy):
- Conservative: only output a match if confidence is high
- Entity-level gate: reject all matches for an S1 if its best score is too low
- Score margin: flag ambiguous cases for stricter treatment
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Applies the final decision policy to convert candidate scores into
    binary match predictions.

    Usage::

        engine = DecisionEngine(config)
        engine.set_thresholds(threshold, s2_threshold, s3_threshold)
        predictions = engine.decide(pair_scores)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.threshold: float = config.get("default_threshold", 0.75)
        self.s2_threshold: float = self.threshold
        self.s3_threshold: float = self.threshold
        self.min_entity_score: float = config.get("min_entity_score", 0.40)
        self.margin_threshold: float = config.get("margin_threshold", 0.15)
        self.use_source_specific: bool = config.get("source_specific_thresholds", True)

    def set_thresholds(
        self,
        threshold: float,
        s2_threshold: Optional[float] = None,
        s3_threshold: Optional[float] = None,
    ):
        """Update decision thresholds after optimization."""
        self.threshold = threshold
        self.s2_threshold = s2_threshold if s2_threshold is not None else threshold
        self.s3_threshold = s3_threshold if s3_threshold is not None else threshold
        logger.info(
            "Thresholds set — global: %.3f, S2: %.3f, S3: %.3f",
            self.threshold, self.s2_threshold, self.s3_threshold,
        )

    def decide(
        self,
        pair_scores: Dict[str, Dict[str, float]],
        s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Convert candidate scores into final binary match predictions.

        Decision pipeline per S1 entity:
        1. Entity-level gate: if max_score < min_entity_score → empty prediction.
        2. Score margin check: if margin is tiny (two candidates very close),
           apply stricter threshold.
        3. Per-candidate threshold: keep candidates above source-specific threshold.

        Args:
            pair_scores: {s1_id -> {s23_id -> probability score}}
            s1_ids: All S1 entity IDs (to ensure every ID gets a row).

        Returns:
            {s1_id -> [matched_entity_ids]} (empty list = singleton/no match)
        """
        if s1_ids is None:
            s1_ids = list(pair_scores.keys())

        predictions: Dict[str, List[str]] = {}

        for s1_id in s1_ids:
            scores = pair_scores.get(s1_id, {})

            if not scores:
                # No candidates at all → no match
                predictions[s1_id] = []
                continue

            sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_score = sorted_candidates[0][1]
            second_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
            score_margin = top_score - second_score

            # Entity-level gate
            if top_score < self.min_entity_score:
                predictions[s1_id] = []
                continue

            # Determine effective threshold
            # If margin is very small (ambiguous), we raise the bar
            effective_threshold = self.threshold
            if score_margin < self.margin_threshold and top_score < 0.95:
                effective_threshold = min(self.threshold + 0.1, 0.98)

            # Per-candidate filtering with source-specific thresholds
            accepted = []
            for s23_id, score in sorted_candidates:
                if self.use_source_specific:
                    thr = (
                        self.s2_threshold if s23_id.startswith("S2-")
                        else self.s3_threshold
                    )
                    # Use max of source-specific and effective (margin-adjusted)
                    thr = max(thr, effective_threshold)
                else:
                    thr = effective_threshold

                if score >= thr:
                    accepted.append(s23_id)

            predictions[s1_id] = accepted

        return predictions

    def decide_with_stats(
        self,
        pair_scores: Dict[str, Dict[str, float]],
        s1_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Decide and also return statistics about the decision.

        Returns:
            (predictions, stats)
        """
        predictions = self.decide(pair_scores, s1_ids)

        if s1_ids is None:
            s1_ids = list(pair_scores.keys())

        stats = {
            "n_s1_entities": len(s1_ids),
            "n_with_predictions": sum(1 for v in predictions.values() if v),
            "n_empty_predictions": sum(1 for v in predictions.values() if not v),
            "n_multi_match": sum(1 for v in predictions.values() if len(v) > 1),
            "avg_matches_per_s1": float(
                np.mean([len(v) for v in predictions.values()])
            ) if predictions else 0.0,
            "max_matches_per_s1": max((len(v) for v in predictions.values()), default=0),
        }

        logger.info(
            "Decision stats: %d S1 entities, %d with matches, %d empty, %d multi-match",
            stats["n_s1_entities"],
            stats["n_with_predictions"],
            stats["n_empty_predictions"],
            stats["n_multi_match"],
        )

        return predictions, stats


# ---------------------------------------------------------------------------
# Policy alternatives (for experimentation)
# ---------------------------------------------------------------------------

def policy_top_k(
    pair_scores: Dict[str, Dict[str, float]],
    k: int = 1,
    min_score: float = 0.5,
) -> Dict[str, List[str]]:
    """
    Policy: Take top-K candidates above min_score.

    Useful as a conservative baseline.
    """
    predictions = {}
    for s1_id, scores in pair_scores.items():
        sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predictions[s1_id] = [
            mid for mid, sc in sorted_c[:k] if sc >= min_score
        ]
    return predictions


def policy_threshold(
    pair_scores: Dict[str, Dict[str, float]],
    threshold: float = 0.75,
) -> Dict[str, List[str]]:
    """
    Simple threshold policy: accept all above threshold.
    """
    return {
        s1_id: [mid for mid, sc in scores.items() if sc >= threshold]
        for s1_id, scores in pair_scores.items()
    }


def policy_adaptive(
    pair_scores: Dict[str, Dict[str, float]],
    base_threshold: float = 0.75,
    min_entity_score: float = 0.40,
    margin_threshold: float = 0.15,
    margin_penalty: float = 0.10,
) -> Dict[str, List[str]]:
    """
    Adaptive policy: entity-level gate + margin-based threshold adjustment.
    """
    predictions = {}
    for s1_id, scores in pair_scores.items():
        if not scores:
            predictions[s1_id] = []
            continue

        sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = sorted_c[0][1]

        if top_score < min_entity_score:
            predictions[s1_id] = []
            continue

        second_score = sorted_c[1][1] if len(sorted_c) > 1 else 0.0
        margin = top_score - second_score

        thr = base_threshold
        if margin < margin_threshold and top_score < 0.95:
            thr = base_threshold + margin_penalty

        predictions[s1_id] = [mid for mid, sc in sorted_c if sc >= thr]

    return predictions
