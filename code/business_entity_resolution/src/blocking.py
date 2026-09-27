"""
blocking.py — Multi-channel candidate generation (blocking) engine.

Implements high-recall, bounded-complexity blocking strategies:
  A. Exact cleaned name (within country)
  B. Exact stripped name (within country)
  C. Exact sorted-token name (within country)
  D. Exact alphanumeric name (within country)
  E. Postal code candidate (within country)
  F. First numeric (house number) + first name token (within country)
  G. Informative name tokens with bounded bucket size (within country)
  H. Informative address tokens with bounded bucket size (within country)

All channels are strictly partitioned by country, guaranteeing 0% cross-country leakage.
Candidates are ranked by accumulated channel evidence and capped to top_k candidates per S1 entity.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BlockingEngine:
    """
    High-speed, memory-efficient multi-channel candidate generation engine.

    Usage::

        engine = BlockingEngine(config)
        engine.fit(s2_df, s3_df)          # build indices on S2+S3
        candidates = engine.get_candidates(s1_df)
        # candidates: Dict[s1_id -> Set[s2/s3_id]]
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or {}
        self.top_k_candidates: int = self.cfg.get("top_k_candidates", 40)
        self.min_token_len: int = self.cfg.get("min_token_len", 3)
        self.max_bucket_size: int = self.cfg.get("max_bucket_size", 120)

        # Inverted index: (country, channel, value) -> List[int]  (integer positions into _s23_ids)
        # Using ints instead of strings cuts memory ~5x for large corpora.
        self._inv_index: Dict[Tuple, List[int]] = defaultdict(list)
        self._token_doc_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        self._s23_ids: List[str] = []   # index -> entity_id lookup table

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "BlockingEngine":
        """
        Build all blocking indices on the S2+S3 candidate pool.
        Stores integer positions in the inverted index to minimise RAM.
        """
        s23 = pd.concat([s2_df, s3_df], ignore_index=True)
        self._s23_ids = s23["entity_id"].tolist()
        n = len(s23)
        logger.info("BlockingEngine.fit — S23 pool size: %d", n)

        # Fast extraction via native lists (no full dict materialisation)
        eids_idx  = list(range(n))   # integer positions — avoids storing strings in index
        countries = s23.get("country_norm", s23.get("country", "")).fillna("").astype(str).str.lower().tolist()
        names_clean   = s23.get("name_cleaned",        "").fillna("").tolist()
        names_strip   = s23.get("name_stripped",       "").fillna("").tolist()
        names_sort    = s23.get("name_sorted_tokens",  "").fillna("").tolist()
        names_alnum   = s23.get("name_alphanumeric",   "").fillna("").tolist()
        names_tok_str = s23.get("name_tokens",         "").fillna("").tolist()
        addrs_tok_str = s23.get("addr_tokens",         "").fillna("").tolist()
        postals       = s23.get("addr_postal_str",     "").fillna("").tolist()
        first_nums    = s23.get("addr_first_numeric",  "").fillna("").tolist()

        # Step 1: Compute document frequencies for tokens per country
        self._token_doc_freq.clear()
        for c, nt_str, at_str in zip(countries, names_tok_str, addrs_tok_str):
            toks = set(nt_str.split() + at_str.split())
            for t in toks:
                if len(t) >= self.min_token_len:
                    self._token_doc_freq[(c, t)] += 1

        # Step 2: Build inverted index (integer positions, bounded bucket size)
        self._inv_index.clear()
        max_b  = self.max_bucket_size
        min_tl = self.min_token_len
        idx    = self._inv_index

        for i, c, nc, ns, nso, na, nt_str, at_str, pos, fn in zip(
            eids_idx, countries, names_clean, names_strip, names_sort,
            names_alnum, names_tok_str, addrs_tok_str, postals, first_nums
        ):
            if nc:
                bucket = idx[(c, "nc", nc)]
                if len(bucket) < max_b: bucket.append(i)
            if ns:
                bucket = idx[(c, "ns", ns)]
                if len(bucket) < max_b: bucket.append(i)
            if nso:
                bucket = idx[(c, "nso", nso)]
                if len(bucket) < max_b: bucket.append(i)
            if na and len(na) >= 4:
                bucket = idx[(c, "na", na)]
                if len(bucket) < max_b: bucket.append(i)
            if pos:
                bucket = idx[(c, "pos", pos)]
                if len(bucket) < max_b: bucket.append(i)

            nt_list = nt_str.split()
            if fn and nt_list:
                bucket = idx[(c, "num_tok", fn, nt_list[0])]
                if len(bucket) < max_b: bucket.append(i)

            for tok in nt_list:
                if len(tok) >= min_tl and self._token_doc_freq[(c, tok)] <= max_b:
                    bucket = idx[(c, "ntok", tok)]
                    if len(bucket) < max_b: bucket.append(i)

            for tok in at_str.split():
                if len(tok) >= min_tl and self._token_doc_freq[(c, tok)] <= max_b:
                    bucket = idx[(c, "atok", tok)]
                    if len(bucket) < max_b: bucket.append(i)

        logger.info("BlockingEngine.fit complete — distinct index keys: %d", len(self._inv_index))
        return self

    def get_candidates(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Generate candidate (S2/S3) entity IDs for each S1 entity.
        Returns:
            Dict mapping s1_entity_id -> set of candidate s2/s3 entity_ids
        """
        candidates: Dict[str, Set[str]] = {}
        s23_ids = self._s23_ids   # integer position -> entity_id string

        eids          = s1_df["entity_id"].tolist()
        countries     = s1_df.get("country_norm", s1_df.get("country", "")).fillna("").astype(str).str.lower().tolist()
        names_clean   = s1_df.get("name_cleaned",       "").fillna("").tolist()
        names_strip   = s1_df.get("name_stripped",      "").fillna("").tolist()
        names_sort    = s1_df.get("name_sorted_tokens", "").fillna("").tolist()
        names_alnum   = s1_df.get("name_alphanumeric",  "").fillna("").tolist()
        names_tok_str = s1_df.get("name_tokens",        "").fillna("").tolist()
        addrs_tok_str = s1_df.get("addr_tokens",        "").fillna("").tolist()
        postals       = s1_df.get("addr_postal_str",    "").fillna("").tolist()
        first_nums    = s1_df.get("addr_first_numeric", "").fillna("").tolist()

        idx   = self._inv_index
        freq  = self._token_doc_freq
        top_k = self.top_k_candidates
        max_b = self.max_bucket_size
        min_tl = self.min_token_len

        for eid, c, nc, ns, nso, na, nt_str, at_str, pos, fn in zip(
            eids, countries, names_clean, names_strip, names_sort,
            names_alnum, names_tok_str, addrs_tok_str, postals, first_nums
        ):
            scores: Dict[int, int] = defaultdict(int)

            if nc:
                for i in idx.get((c, "nc", nc), []):
                    scores[i] += 8
            if ns:
                for i in idx.get((c, "ns", ns), []):
                    scores[i] += 6
            if nso:
                for i in idx.get((c, "nso", nso), []):
                    scores[i] += 5
            if na and len(na) >= 4:
                for i in idx.get((c, "na", na), []):
                    scores[i] += 5
            if pos:
                for i in idx.get((c, "pos", pos), []):
                    scores[i] += 2

            nt_list = nt_str.split()
            if fn and nt_list:
                for i in idx.get((c, "num_tok", fn, nt_list[0]), []):
                    scores[i] += 3

            for tok in nt_list:
                if len(tok) >= min_tl and freq[(c, tok)] <= max_b:
                    for i in idx.get((c, "ntok", tok), []):
                        scores[i] += 2

            for tok in at_str.split():
                if len(tok) >= min_tl and freq[(c, tok)] <= max_b:
                    for i in idx.get((c, "atok", tok), []):
                        scores[i] += 1

            if scores:
                top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                # Convert integer positions back to entity_id strings
                candidates[eid] = {s23_ids[i] for i, _ in top_items}
            else:
                candidates[eid] = set()

        total = sum(len(v) for v in candidates.values())
        avg   = total / max(len(candidates), 1)
        max_c = max((len(v) for v in candidates.values()), default=0)
        logger.info(
            "Blocking complete — total candidates: %d, avg per S1: %.1f, max: %d",
            total, avg, max_c,
        )
        return candidates

    def evaluate_blocking(
        self,
        candidates: Dict[str, Set[str]],
        gt_dict: Dict[str, List[str]],
        s1_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute blocking recall and statistics.
        """
        total_true = 0
        true_in_cands = 0
        missed_pairs = []

        eval_ids = s1_ids if s1_ids is not None else list(gt_dict.keys())

        for s1_id in eval_ids:
            true_matches = gt_dict.get(s1_id, [])
            cands = candidates.get(s1_id, set())
            for mid in true_matches:
                total_true += 1
                if mid in cands:
                    true_in_cands += 1
                else:
                    missed_pairs.append((s1_id, mid))

        total_cands = sum(len(candidates.get(sid, set())) for sid in eval_ids)
        avg_cands = total_cands / max(len(eval_ids), 1)
        max_cands = max((len(candidates.get(sid, set())) for sid in eval_ids), default=0)
        recall = true_in_cands / max(total_true, 1)

        return {
            "blocking_recall": recall,
            "true_matches_total": total_true,
            "true_matches_in_candidates": true_in_cands,
            "candidate_count_total": total_cands,
            "avg_candidates_per_s1": avg_cands,
            "max_candidates_per_s1": max_cands,
            "missed_pair_count": len(missed_pairs),
            "missed_pairs_sample": missed_pairs[:20],
        }
