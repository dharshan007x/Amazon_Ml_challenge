"""
features.py — Rich pairwise feature engineering for entity resolution.

Feature groups:
  NAME   : ~22 features (exact, edit, token, char-ngram, TF-IDF, rare-token)
  ADDRESS: ~15 features (exact, token, numeric, postal, rare-token)
  COUNTRY: ~4 features
  COMBINED: ~8 cross-group features
  MISSINGNESS: ~6 features
  SOURCE: ~2 features
  BLOCKING: ~2 features (if enabled)

Total: ~60 features per candidate pair.

Design:
- FeatureEngine is fitted once on the combined corpus (train S1+S2+S3 or test S1+S2+S3)
- Caches TF-IDF matrices keyed by entity_id for O(1) retrieval per pair
- Uses rapidfuzz for fast string similarity computation
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    from rapidfuzz import distance as rf_dist
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logging.getLogger(__name__).warning(
        "rapidfuzz not available — edit distance features will be 0. Install: pip install rapidfuzz"
    )

logger = logging.getLogger(__name__)

# Legal suffix set for fast stripped-token construction in flat-column fallback
# (mirrors normalization.LEGAL_SUFFIX_SET without circular import)
_LEGAL_SUFFIX_SET_FEAT: Set[str] = {
    "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llc", "inc", "incorporated", "lp", "llp",
    "plc", "gmbh", "sarl", "sas", "sa", "ag", "bv", "nv",
    "pty", "pte", "sdn", "bhd", "kk", "oy", "ab",
}


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Computes pairwise features for candidate (S1, S2/S3) entity pairs.

    Lifecycle::

        engine = FeatureEngine(config)
        engine.fit(all_dfs)        # fit TF-IDF, compute IDF weights
        X = engine.compute_batch(pairs_df, s1_lookup, s23_lookup)
        names = engine.feature_names()
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.ngram_sizes: List[int] = config.get("ngram_sizes", [3, 4])
        self.use_cross_source: bool = config.get("use_cross_source", True)
        self.use_blocking_evidence: bool = config.get("use_blocking_evidence", True)

        # TF-IDF vectorizers (name and address, word and char)
        self._name_word_vec: Optional[TfidfVectorizer] = None
        self._name_char_vec: Optional[TfidfVectorizer] = None
        self._addr_word_vec: Optional[TfidfVectorizer] = None

        # TF-IDF matrices (n_entities x vocab), keyed by entity_id -> row index
        self._name_word_mat = None
        self._name_char_mat = None
        self._addr_word_mat = None
        self._eid_to_row: Dict[str, int] = {}

        # Token IDF weights
        self._name_token_idf: Dict[str, float] = {}
        self._addr_token_idf: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(self, dfs: List[pd.DataFrame]) -> "FeatureEngine":
        """
        Fit TF-IDF vectorizers and compute token IDF weights on the combined corpus.

        Args:
            dfs: List of normalized DataFrames (all sources involved in this split).
        """
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("FeatureEngine.fit on %d entities.", len(combined))

        all_ids = combined["entity_id"].tolist()
        self._eid_to_row = {eid: i for i, eid in enumerate(all_ids)}

        name_corpus = combined["name_cleaned"].fillna("").tolist()
        addr_corpus = combined["addr_expanded"].fillna("").tolist()

        # Word-level name TF-IDF
        self._name_word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.97, sublinear_tf=True
        )
        mat = self._name_word_vec.fit_transform(name_corpus)
        self._name_word_mat = normalize(mat, norm="l2", copy=False)

        # Char-ngram name TF-IDF
        self._name_char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=1, max_df=0.98, sublinear_tf=True
        )
        mat = self._name_char_vec.fit_transform(name_corpus)
        self._name_char_mat = normalize(mat, norm="l2", copy=False)

        # Word-level address TF-IDF
        self._addr_word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.97, sublinear_tf=True
        )
        mat = self._addr_word_vec.fit_transform(addr_corpus)
        self._addr_word_mat = normalize(mat, norm="l2", copy=False)

        # Token IDF weights (name)
        n = len(combined)
        name_df: Dict[str, int] = {}
        for tok_str in combined["name_tokens"].fillna(""):
            for t in set(tok_str.split()):
                name_df[t] = name_df.get(t, 0) + 1
        self._name_token_idf = {
            t: np.log((n + 1) / (df + 1)) + 1.0 for t, df in name_df.items()
        }

        # Token IDF weights (address)
        addr_df: Dict[str, int] = {}
        for tok_str in combined["addr_tokens"].fillna(""):
            for t in set(tok_str.split()):
                addr_df[t] = addr_df.get(t, 0) + 1
        self._addr_token_idf = {
            t: np.log((n + 1) / (df + 1)) + 1.0 for t, df in addr_df.items()
        }

        logger.info("FeatureEngine.fit complete.")
        return self

    # -----------------------------------------------------------------------
    # Single-pair feature computation
    # -----------------------------------------------------------------------

    def compute_features(
        self,
        s1_row: dict,
        s23_row: dict,
        blocking_channels: int = 1,
    ) -> np.ndarray:
        """
        Compute all features for a single (S1, S23) candidate pair.

        Args:
            s1_row: Dict from normalized S1 DataFrame row.
            s23_row: Dict from normalized S2/S3 DataFrame row.
            blocking_channels: Number of blocking channels that generated this pair.

        Returns:
            1-D numpy array of features.
        """
        feats: Dict[str, float] = {}

        nv1 = s1_row.get("_name_views") or {}
        nv2 = s23_row.get("_name_views") or {}
        av1 = s1_row.get("_addr_views") or {}
        av2 = s23_row.get("_addr_views") or {}

        # Fast fallback: reconstruct view fields directly from flat columns
        # (avoids expensive normalize_name/normalize_address calls per pair)
        if not nv1:
            nc1 = s1_row.get("name_cleaned", "")
            toks1 = nc1.split()
            stok1 = [t for t in toks1 if t not in _LEGAL_SUFFIX_SET_FEAT]
            nv1 = {
                "cleaned": nc1,
                "stripped": s1_row.get("name_stripped", ""),
                "sorted_tokens": s1_row.get("name_sorted_tokens", ""),
                "stripped_sorted": s1_row.get("name_stripped_sorted", ""),
                "alphanumeric": s1_row.get("name_alphanumeric", ""),
                "token_set": set(toks1),
                "stripped_token_set": set(stok1),
            }
        if not nv2:
            nc2 = s23_row.get("name_cleaned", "")
            toks2 = nc2.split()
            stok2 = [t for t in toks2 if t not in _LEGAL_SUFFIX_SET_FEAT]
            nv2 = {
                "cleaned": nc2,
                "stripped": s23_row.get("name_stripped", ""),
                "sorted_tokens": s23_row.get("name_sorted_tokens", ""),
                "stripped_sorted": s23_row.get("name_stripped_sorted", ""),
                "alphanumeric": s23_row.get("name_alphanumeric", ""),
                "token_set": set(toks2),
                "stripped_token_set": set(stok2),
            }
        if not av1:
            av1 = {
                "cleaned": s1_row.get("addr_cleaned", ""),
                "expanded": s1_row.get("addr_expanded", ""),
                "tokens": s1_row.get("addr_tokens", ""),
                "sorted_tokens": s1_row.get("addr_sorted_tokens", ""),
                "token_set": set(s1_row.get("addr_tokens", "").split()),
                "first_numeric": s1_row.get("addr_first_numeric", ""),
                "postal_candidates": [s1_row["addr_postal_str"]] if s1_row.get("addr_postal_str") else [],
                "numeric_tokens": [],
            }
        if not av2:
            av2 = {
                "cleaned": s23_row.get("addr_cleaned", ""),
                "expanded": s23_row.get("addr_expanded", ""),
                "tokens": s23_row.get("addr_tokens", ""),
                "sorted_tokens": s23_row.get("addr_sorted_tokens", ""),
                "token_set": set(s23_row.get("addr_tokens", "").split()),
                "first_numeric": s23_row.get("addr_first_numeric", ""),
                "postal_candidates": [s23_row["addr_postal_str"]] if s23_row.get("addr_postal_str") else [],
                "numeric_tokens": [],
            }

        # ======================
        # NAME FEATURES
        # ======================
        n_clean1 = nv1.get("cleaned", "")
        n_clean2 = nv2.get("cleaned", "")
        n_strip1 = nv1.get("stripped", "")
        n_strip2 = nv2.get("stripped", "")
        n_sort1 = nv1.get("sorted_tokens", "")
        n_sort2 = nv2.get("sorted_tokens", "")
        n_ssort1 = nv1.get("stripped_sorted", "")
        n_ssort2 = nv2.get("stripped_sorted", "")
        t1: Set[str] = nv1.get("token_set", set())
        t2: Set[str] = nv2.get("token_set", set())
        s1: Set[str] = nv1.get("stripped_token_set", set())
        s2: Set[str] = nv2.get("stripped_token_set", set())

        # Exact matches
        feats["name_exact_cleaned"] = float(n_clean1 == n_clean2 and n_clean1 != "")
        feats["name_exact_stripped"] = float(n_strip1 == n_strip2 and n_strip1 != "")
        feats["name_exact_sorted"] = float(n_sort1 == n_sort2 and n_sort1 != "")
        feats["name_exact_stripped_sorted"] = float(n_ssort1 == n_ssort2 and n_ssort1 != "")

        # Edit similarity (Levenshtein / Jaro-Winkler)
        if _HAS_RAPIDFUZZ:
            feats["name_lev_sim"] = rf_dist.Levenshtein.normalized_similarity(n_clean1, n_clean2)
            feats["name_lev_sim_stripped"] = rf_dist.Levenshtein.normalized_similarity(n_strip1, n_strip2)
            feats["name_jaro_winkler"] = rf_dist.JaroWinkler.normalized_similarity(n_clean1, n_clean2)
            feats["name_token_sort_ratio"] = rf_fuzz.token_sort_ratio(n_clean1, n_clean2) / 100.0
            feats["name_token_set_ratio"] = rf_fuzz.token_set_ratio(n_clean1, n_clean2) / 100.0
        else:
            feats["name_lev_sim"] = 0.0
            feats["name_lev_sim_stripped"] = 0.0
            feats["name_jaro_winkler"] = 0.0
            feats["name_token_sort_ratio"] = 0.0
            feats["name_token_set_ratio"] = 0.0

        # Token-set features
        t_inter = t1 & t2
        t_union = t1 | t2
        feats["name_token_jaccard"] = len(t_inter) / max(len(t_union), 1)
        feats["name_token_overlap"] = len(t_inter) / max(min(len(t1), len(t2)), 1)
        feats["name_token_contain_1in2"] = len(t_inter) / max(len(t1), 1)
        feats["name_token_contain_2in1"] = len(t_inter) / max(len(t2), 1)

        # Stripped token features
        s_inter = s1 & s2
        s_union = s1 | s2
        feats["name_stripped_token_jaccard"] = len(s_inter) / max(len(s_union), 1)

        # Rare-token / IDF-weighted Jaccard
        idf1_sum = sum(self._name_token_idf.get(t, 1.0) for t in t1)
        idf2_sum = sum(self._name_token_idf.get(t, 1.0) for t in t2)
        idf_common = sum(self._name_token_idf.get(t, 1.0) for t in t_inter)
        denom = idf1_sum + idf2_sum - idf_common
        feats["name_weighted_jaccard"] = idf_common / max(denom, 1e-6)
        feats["name_rare_token_overlap"] = idf_common / max(min(idf1_sum, idf2_sum), 1e-6)

        # Length features
        l1, l2 = len(n_clean1), len(n_clean2)
        feats["name_length_ratio"] = min(l1, l2) / max(max(l1, l2), 1)
        feats["name_length_diff"] = float(abs(l1 - l2))

        # Prefix ratio
        min_len = min(l1, l2)
        pref = 0
        for ci in range(min_len):
            if n_clean1[ci] == n_clean2[ci]:
                pref += 1
            else:
                break
        feats["name_prefix_ratio"] = pref / max(min_len, 1)

        # TF-IDF cosine (pre-computed matrices)
        eid1 = s1_row.get("entity_id", "")
        eid2 = s23_row.get("entity_id", "")
        feats["name_tfidf_word_cos"] = self._cosine(eid1, eid2, self._name_word_mat)
        feats["name_tfidf_char_cos"] = self._cosine(eid1, eid2, self._name_char_mat)

        # ======================
        # ADDRESS FEATURES
        # ======================
        a_clean1 = av1.get("expanded", "")
        a_clean2 = av2.get("expanded", "")
        at1: Set[str] = av1.get("token_set", set())
        at2: Set[str] = av2.get("token_set", set())

        feats["addr_exact_match"] = float(a_clean1 == a_clean2 and a_clean1 != "")

        at_inter = at1 & at2
        at_union = at1 | at2
        feats["addr_token_jaccard"] = len(at_inter) / max(len(at_union), 1)
        feats["addr_token_overlap"] = len(at_inter) / max(min(len(at1), len(at2)), 1)
        feats["addr_token_contain"] = len(at_inter) / max(len(at1), 1)

        # Numeric / structural
        num1 = set(av1.get("numeric_tokens", []))
        num2 = set(av2.get("numeric_tokens", []))
        num_inter = num1 & num2
        num_union = num1 | num2
        feats["addr_numeric_jaccard"] = len(num_inter) / max(len(num_union), 1)
        feats["addr_numeric_overlap"] = len(num_inter) / max(min(len(num1), len(num2)), 1)

        fn1 = av1.get("first_numeric", "")
        fn2 = av2.get("first_numeric", "")
        feats["addr_house_number_match"] = float(fn1 != "" and fn1 == fn2)

        pc1 = set(av1.get("postal_candidates", []))
        pc2 = set(av2.get("postal_candidates", []))
        feats["addr_postal_code_match"] = float(bool(pc1 & pc2))
        feats["addr_postal_code_jaccard"] = len(pc1 & pc2) / max(len(pc1 | pc2), 1)

        # IDF-weighted address Jaccard
        aidf1 = sum(self._addr_token_idf.get(t, 1.0) for t in at1)
        aidf2 = sum(self._addr_token_idf.get(t, 1.0) for t in at2)
        aidf_common = sum(self._addr_token_idf.get(t, 1.0) for t in at_inter)
        adenom = aidf1 + aidf2 - aidf_common
        feats["addr_weighted_jaccard"] = aidf_common / max(adenom, 1e-6)

        # Address length ratio
        al1 = av1.get("char_count", 0)
        al2 = av2.get("char_count", 0)
        feats["addr_length_ratio"] = min(al1, al2) / max(max(al1, al2), 1)

        # TF-IDF cosine address
        feats["addr_tfidf_word_cos"] = self._cosine(eid1, eid2, self._addr_word_mat)

        # ======================
        # COUNTRY FEATURES
        # ======================
        c1 = s1_row.get("country_norm", "")
        c2 = s23_row.get("country_norm", "")
        feats["country_exact_match"] = float(c1 == c2 and c1 != "")
        feats["country_missing_s1"] = float(c1 == "")
        feats["country_missing_s2"] = float(c2 == "")
        feats["country_both_missing"] = float(c1 == "" and c2 == "")

        # ======================
        # COMBINED FEATURES
        # ======================
        best_name = max(
            feats["name_token_jaccard"],
            feats["name_weighted_jaccard"],
            feats.get("name_lev_sim", 0.0),
            feats.get("name_tfidf_word_cos", 0.0),
        )
        best_addr = max(
            feats["addr_token_jaccard"],
            feats["addr_weighted_jaccard"],
            feats.get("addr_tfidf_word_cos", 0.0),
        )
        feats["combined_name_x_addr"] = best_name * best_addr
        feats["combined_name_plus_addr"] = (best_name + best_addr) / 2.0
        feats["combined_max"] = max(best_name, best_addr)
        feats["combined_min"] = min(best_name, best_addr)
        feats["high_name_low_addr"] = float(best_name >= 0.7 and best_addr < 0.3)
        feats["low_name_high_addr"] = float(best_name < 0.3 and best_addr >= 0.7)
        feats["both_high"] = float(best_name >= 0.7 and best_addr >= 0.7)
        feats["country_x_name"] = feats["country_exact_match"] * best_name
        feats["country_x_addr"] = feats["country_exact_match"] * best_addr

        # ======================
        # MISSINGNESS FEATURES
        # ======================
        feats["name_missing_s1"] = float(n_clean1 == "")
        feats["name_missing_s2"] = float(n_clean2 == "")
        feats["addr_missing_s1"] = float(a_clean1 == "")
        feats["addr_missing_s2"] = float(a_clean2 == "")
        feats["both_name_missing"] = float(n_clean1 == "" and n_clean2 == "")
        feats["both_addr_missing"] = float(a_clean1 == "" and a_clean2 == "")

        # ======================
        # SOURCE FEATURES
        # ======================
        source_str = s23_row.get("entity_id", "")
        feats["is_s2_pair"] = float(source_str.startswith("S2-"))
        feats["is_s3_pair"] = float(source_str.startswith("S3-"))

        # ======================
        # BLOCKING EVIDENCE
        # ======================
        if self.use_blocking_evidence:
            feats["blocking_channels"] = float(blocking_channels)

        return np.array(list(feats.values()), dtype=np.float32), list(feats.keys())

    # -----------------------------------------------------------------------
    # Batch feature computation
    # -----------------------------------------------------------------------

    def compute_batch(
        self,
        pairs: List[Tuple[str, str]],   # [(s1_id, s23_id), ...]
        s1_lookup: Dict[str, dict],     # entity_id -> normalized row dict
        s23_lookup: Dict[str, dict],    # entity_id -> normalized row dict
        channel_counts: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute features for a list of candidate pairs.

        Returns:
            (X, feature_names)
            X shape: (n_pairs, n_features)
        """
        feature_names: Optional[List[str]] = None
        rows = []

        for s1_id, s23_id in pairs:
            s1_row = s1_lookup.get(s1_id, {})
            s23_row = s23_lookup.get(s23_id, {})
            n_channels = channel_counts.get((s1_id, s23_id), 1) if channel_counts else 1

            feat_arr, feat_names = self.compute_features(s1_row, s23_row, n_channels)
            rows.append(feat_arr)
            if feature_names is None:
                feature_names = feat_names

        if not rows:
            return np.zeros((0, 0), dtype=np.float32), feature_names or []

        X = np.stack(rows, axis=0)
        return X, feature_names

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _cosine(self, eid1: str, eid2: str, mat) -> float:
        """Compute cosine similarity between two entities from a precomputed TF-IDF matrix."""
        if mat is None:
            return 0.0
        r1 = self._eid_to_row.get(eid1, -1)
        r2 = self._eid_to_row.get(eid2, -1)
        if r1 < 0 or r2 < 0:
            return 0.0
        # Dot product of L2-normalized rows
        sim = float((mat[r1] @ mat[r2].T).toarray()[0, 0])
        return max(0.0, min(1.0, sim))

    def feature_names(self) -> List[str]:
        """Return the list of feature names (in order)."""
        # Compute a dummy pair to get names
        dummy = {"entity_id": "", "business_name": "", "business_address": "",
                 "country_norm": "", "name_cleaned": "", "name_tokens": "",
                 "name_sorted_tokens": "", "name_stripped": "", "name_stripped_sorted": "",
                 "addr_expanded": "", "addr_tokens": "", "addr_first_numeric": "",
                 "addr_postal_str": "", "_name_views": {}, "_addr_views": {}}
        _, names = self.compute_features(dummy, dummy)
        return names
