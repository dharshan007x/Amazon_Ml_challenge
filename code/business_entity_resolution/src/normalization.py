"""
normalization.py — Multi-view text normalization for business names and addresses.

Creates 11+ views for business names and 14+ views for addresses.
All normalization is purely local (no external geocoder or API).

Key design choices:
- Legal suffixes are IDENTIFIED but kept as a SEPARATE view (not silently deleted)
- Address abbreviations are EXPANDED for better matching
- Unicode diacritics are removed for cross-language robustness
- Rare tokens (computed from corpus IDF) are tracked for weighted similarity
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Legal business suffix tokens (lowercase)
LEGAL_SUFFIX_SET: Set[str] = {
    "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llc", "inc", "incorporated", "lp", "llp",
    "plc", "gmbh", "sarl", "sas", "sa", "ag", "bv", "nv",
    "pty", "pte", "sdn", "bhd", "kk", "oy", "ab",
}

# Address expansion patterns: abbreviated form -> expanded form
# Applied AFTER lowercasing; spaces around boundary ensure word-level match
ADDR_EXPANSION_PATTERNS: List[Tuple[str, str]] = [
    (r"\brd\b", "road"),
    (r"\bst\b", "street"),
    (r"\bave\b", "avenue"),
    (r"\blane\b", "lane"),
    (r"\bln\b", "lane"),
    (r"\bblvd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bct\b", "court"),
    (r"\bpl\b", "place"),
    (r"\bsq\b", "square"),
    (r"\bhwy\b", "highway"),
    (r"\bexpy\b", "expressway"),
    (r"\bn\b", "north"),
    (r"\bs\b", "south"),
    (r"\be\b", "east"),
    (r"\bw\b", "west"),
    (r"\bne\b", "northeast"),
    (r"\bnw\b", "northwest"),
    (r"\bse\b", "southeast"),
    (r"\bsw\b", "southwest"),
    (r"\bapt\b", "apartment"),
    (r"\bste\b", "suite"),
    (r"\bfl\b", "floor"),
    (r"\bflr\b", "floor"),
    (r"\bft\b", "fort"),
    (r"\bmt\b", "mount"),
]

# Punctuation / separator characters to normalise to space
_PUNCT_RE = re.compile(r"[,;|/\\&!?@#$%^*()\[\]{}<>\"'`~=+]")
_DASH_RE = re.compile(r"[-–—]")
_DOT_EXCEPT_DECIMAL_RE = re.compile(r"(?<!\d)\.(?!\d)")
_AND_RE = re.compile(r"\b&\b")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Postal code patterns (US, India, France, generic)
_POSTAL_PATTERNS = [
    re.compile(r"\b\d{5}(?:-\d{4})?\b"),  # US ZIP
    re.compile(r"\b\d{6}\b"),             # India PIN
    re.compile(r"\b\d{2}[ ]?\d{3}\b"),    # France
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"),  # UK
    re.compile(r"\b\d{4,6}\b"),           # generic 4-6 digit
]

# Numeric sequence extractor
_NUMERIC_RE = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# Compiled regex for stripping combining diacritical marks after NFKD decomposition.
# Much faster than the character-by-character unicodedata.combining() loop.
_COMBINING_RE = re.compile(r"[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


def _unicode_normalize(text: str) -> str:
    """Apply NFKD Unicode normalization."""
    return unicodedata.normalize("NFKD", text)


def _remove_diacritics(text: str) -> str:
    """Strip diacritical marks (ã→a, é→e, etc.) after NFKD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return _COMBINING_RE.sub("", nfkd)


def _normalize_punct(text: str) -> str:
    """Replace common punctuation with spaces; handle & → 'and'."""
    text = _AND_RE.sub(" and ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _DASH_RE.sub(" ", text)
    text = _DOT_EXCEPT_DECIMAL_RE.sub(" ", text)
    return text


def _collapse_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _base_clean(text: str) -> str:
    """Full base cleaning pipeline: unicode → diacritics → lower → punct → whitespace."""
    if not text:
        return ""
    text = _unicode_normalize(text)
    text = _remove_diacritics(text)
    text = text.lower()
    text = _normalize_punct(text)
    text = _collapse_whitespace(text)
    return text


def _tokenize(text: str, min_len: int = 1) -> List[str]:
    """Whitespace tokenize with optional minimum token length."""
    return [t for t in text.split() if len(t) >= min_len]


def _sorted_tokens(tokens: List[str]) -> str:
    return " ".join(sorted(tokens))


def _strip_legal_suffixes(tokens: List[str]) -> List[str]:
    """Remove legal suffix tokens from the token list."""
    return [t for t in tokens if t.lower() not in LEGAL_SUFFIX_SET]


def _alphanumeric_only(text: str) -> str:
    """Keep only alphanumeric characters and spaces."""
    return _collapse_whitespace(re.sub(r"[^a-z0-9\s]", " ", text))


def char_ngrams(text: str, n: int) -> List[str]:
    """Generate character n-grams (with boundary padding)."""
    padded = f" {text} "
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> Dict:
    """
    Generate multiple normalized views of a business name.

    Views produced:
    1.  raw              — original (stripped)
    2.  unicode_norm     — NFKD-normalized
    3.  lowercase        — lowercased only
    4.  punct_norm       — punctuation-normalized
    5.  no_diacritics    — diacritics removed, lowercased
    6.  cleaned          — full pipeline (unicode + diacritics + lower + punct + ws)
    7.  alphanumeric     — alphanumeric chars only
    8.  tokens           — space-joined tokens of cleaned
    9.  sorted_tokens    — space-joined sorted tokens
    10. stripped         — legal-suffix-stripped tokens
    11. stripped_sorted  — stripped + sorted

    Also returns:
    - token_set          — set of tokens from cleaned
    - stripped_token_set — set of tokens after suffix stripping
    """
    raw = "" if (name is None or pd.isna(name)) else str(name).strip()

    cleaned = _base_clean(raw)
    tokens = _tokenize(cleaned)
    sorted_tok = sorted(tokens)
    stripped = _strip_legal_suffixes(tokens)
    stripped_sorted = sorted(stripped)

    return {
        "raw": raw,
        "unicode_norm": _unicode_normalize(raw),
        "lowercase": raw.lower(),
        "punct_norm": _normalize_punct(raw.lower()),
        "no_diacritics": _remove_diacritics(raw.lower()),
        "cleaned": cleaned,
        "alphanumeric": _alphanumeric_only(cleaned),
        "tokens": " ".join(tokens),
        "sorted_tokens": _sorted_tokens(tokens),
        "stripped": " ".join(stripped),
        "stripped_sorted": " ".join(stripped_sorted),
        # Set-valued (for Jaccard etc.)
        "token_set": set(tokens),
        "stripped_token_set": set(stripped),
    }


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

def _expand_address_abbreviations(text: str) -> str:
    """Expand address abbreviations to their full forms."""
    for pattern, replacement in ADDR_EXPANSION_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _extract_numeric_tokens(text: str) -> List[str]:
    """Extract all numeric sequences from text."""
    return _NUMERIC_RE.findall(text)


def _extract_postal_codes(text: str) -> List[str]:
    """Extract postal/zip code candidates."""
    codes = []
    for pat in _POSTAL_PATTERNS:
        codes.extend(m.group().replace(" ", "") for m in pat.finditer(text))
    return list(dict.fromkeys(codes))  # deduplicate, preserve order


def normalize_address(address: str) -> Dict:
    """
    Generate multiple normalized views of a business address.

    Views produced:
    1.  raw              — original (stripped)
    2.  cleaned          — full base clean pipeline
    3.  expanded         — abbreviations expanded, then cleaned
    4.  alphanumeric     — alphanumeric only
    5.  tokens           — space-joined tokens
    6.  sorted_tokens    — space-joined sorted tokens
    7.  token_set        — set of tokens

    Structural fields:
    8.  numeric_tokens   — list of all numeric sequences
    9.  first_numeric    — first numeric sequence (house/building number candidate)
    10. postal_candidates — list of postal-code-like matches
    11. char_count       — character count of cleaned
    12. token_count      — token count
    13. digit_count      — digit character count
    """
    raw = "" if (address is None or pd.isna(address)) else str(address).strip()

    cleaned = _base_clean(raw)
    expanded = _expand_address_abbreviations(cleaned)
    expanded = _collapse_whitespace(expanded)

    tokens = _tokenize(expanded)
    sorted_tok = sorted(tokens)

    numeric_tokens = _extract_numeric_tokens(expanded)
    postal_candidates = _extract_postal_codes(cleaned)  # use cleaned (before expansion)

    return {
        "raw": raw,
        "cleaned": cleaned,
        "expanded": expanded,
        "alphanumeric": _alphanumeric_only(expanded),
        "tokens": " ".join(tokens),
        "sorted_tokens": " ".join(sorted_tok),
        "token_set": set(tokens),
        # Structural
        "numeric_tokens": numeric_tokens,
        "first_numeric": numeric_tokens[0] if numeric_tokens else "",
        "postal_candidates": postal_candidates,
        "char_count": len(expanded),
        "token_count": len(tokens),
        "digit_count": sum(c.isdigit() for c in expanded),
    }


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

def normalize_country(country: str) -> str:
    """Normalize country string to lowercase, diacritics-free."""
    c = "" if (country is None or pd.isna(country)) else str(country).strip().lower()
    c = _remove_diacritics(c)
    c = _collapse_whitespace(c)
    return c


# ---------------------------------------------------------------------------
# DataFrame-level normalization
# ---------------------------------------------------------------------------

import logging as _logging
_norm_logger = _logging.getLogger(__name__)

# Compiled patterns reused across calls
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(sorted(LEGAL_SUFFIX_SET, key=len, reverse=True)) + r")\b"
)


def _vec_base_clean(s: "pd.Series") -> "pd.Series":
    """
    Vectorized version of _base_clean applied to a pandas Series.
    Equivalent to: unicode NFKD → remove diacritics → lower → punct → collapse ws.
    """
    import pandas as _pd
    # NFKD unicode + remove combining chars (diacritics)
    s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    s = s.str.lower()
    # & → and
    s = s.str.replace(r"\b&\b", " and ", regex=True)
    # Punctuation → space
    s = s.str.replace(r"[,;|/\\!?@#$%^*()\[\]{}<>\"'`~=+]", " ", regex=True)
    s = s.str.replace(r"[-\u2013\u2014]", " ", regex=True)
    # Dots not between digits → space
    s = s.str.replace(r"(?<!\d)\.(?!\d)", " ", regex=True)
    # Collapse whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _vec_alphanumeric(s: "pd.Series") -> "pd.Series":
    """Keep only alphanumeric + spaces."""
    return s.str.replace(r"[^a-z0-9\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _vec_strip_suffixes(s: "pd.Series") -> "pd.Series":
    """Remove legal suffix tokens from cleaned name series."""
    return s.str.replace(_LEGAL_SUFFIX_RE, "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _vec_sorted_tokens(s: "pd.Series") -> "pd.Series":
    """Sort whitespace-delimited tokens in each string."""
    return s.apply(lambda x: " ".join(sorted(x.split())) if x else "")


def normalize_dataframe(df: pd.DataFrame, compute_views: bool = True) -> pd.DataFrame:
    """
    Add normalized view columns to a source DataFrame.

    Flat columns (used for blocking & feature lookup) are computed via
    fully vectorized pandas .str operations — fast for millions of rows.

    Args:
        compute_views: If True (default), also compute per-row view dicts
            (_name_views, _addr_views) needed by FeatureEngine.compute_batch.
            Set to False during inference to skip the slow Python loop —
            flat columns are sufficient for blocking and lookup-based features.
    """
    df = df.copy()
    n_rows = len(df)
    _norm_logger.info("normalize_dataframe: %d rows (vectorized)", n_rows)

    name_s = df["business_name"].fillna("")
    addr_s = df["business_address"].fillna("")
    country_s = df["country"].fillna("")

    # ── Name flat columns (fully vectorized) ────────────────────────────
    name_cleaned = _vec_base_clean(name_s)
    name_alnum   = _vec_alphanumeric(name_cleaned)
    name_stripped = _vec_strip_suffixes(name_cleaned)
    name_tokens  = name_cleaned  # space-joined tokens = cleaned string itself
    name_sorted  = _vec_sorted_tokens(name_cleaned)
    name_stripped_sorted = _vec_sorted_tokens(name_stripped)

    df["name_cleaned"]         = name_cleaned
    df["name_tokens"]          = name_tokens
    df["name_sorted_tokens"]   = name_sorted
    df["name_stripped"]        = name_stripped
    df["name_stripped_sorted"] = name_stripped_sorted
    df["name_alphanumeric"]    = name_alnum

    # ── Address flat columns (fully vectorized) ──────────────────────────
    addr_cleaned = _vec_base_clean(addr_s)
    # Expand abbreviations vectorized
    addr_expanded = addr_cleaned.copy()
    for pattern, replacement in ADDR_EXPANSION_PATTERNS:
        addr_expanded = addr_expanded.str.replace(pattern, replacement, regex=True)
    addr_expanded = addr_expanded.str.replace(r"\s+", " ", regex=True).str.strip()

    addr_alnum       = _vec_alphanumeric(addr_expanded)
    addr_tokens      = addr_expanded
    addr_sorted      = _vec_sorted_tokens(addr_expanded)
    addr_first_num   = addr_expanded.str.extract(r"(\d+)", expand=False).fillna("")
    addr_postal      = addr_cleaned.str.extract(
        r"(\b\d{5}(?:-\d{4})?\b|\b\d{6}\b|\b\d{4,6}\b)", expand=False
    ).fillna("")

    df["addr_cleaned"]        = addr_cleaned
    df["addr_expanded"]       = addr_expanded
    df["addr_tokens"]         = addr_tokens
    df["addr_sorted_tokens"]  = addr_sorted
    df["addr_first_numeric"]  = addr_first_num
    df["addr_postal_str"]     = addr_postal

    # ── Country ─────────────────────────────────────────────────────────
    df["country_norm"] = country_s.str.normalize("NFKD") \
        .str.encode("ascii", errors="ignore").str.decode("ascii") \
        .str.lower().str.strip()

    # ── Per-row view dicts (used by FeatureEngine for pairwise features) ─
    if compute_views:
        _norm_logger.info("normalize_dataframe: computing per-row view dicts for %d rows", n_rows)
        name_views = [normalize_name(n) for n in name_s.tolist()]
        addr_views = [normalize_address(a) for a in addr_s.tolist()]
        df["_name_views"] = name_views
        df["_addr_views"] = addr_views
    else:
        _norm_logger.info("normalize_dataframe: skipping per-row view dicts (compute_views=False)")
        df["_name_views"] = None
        df["_addr_views"] = None

    _norm_logger.info("normalize_dataframe: complete (%d rows)", n_rows)
    return df


def get_name_view(df: pd.DataFrame, idx: int) -> Dict:
    """Retrieve cached name views for row `idx`."""
    return df["_name_views"].iloc[idx]


def get_addr_view(df: pd.DataFrame, idx: int) -> Dict:
    """Retrieve cached address views for row `idx`."""
    return df["_addr_views"].iloc[idx]
