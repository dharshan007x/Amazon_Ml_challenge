"""
data.py — Data loading, validation, and audit.

Handles all I/O for the Amazon ML Challenge:
- Loading source TSV files with ID prefix validation
- Parsing ground truth (comma-separated match lists)
- Rich data audit (row counts, nulls, country distribution, match stats)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_tsv(filepath: str) -> pd.DataFrame:
    """Read a TSV file safely, treating all columns as strings."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],   # do NOT treat any value as NaN
    )
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Replace true NaN (from header parsing edge cases) with empty string
    df = df.fillna("")
    return df


def load_source_file(filepath: str, expected_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Load a source TSV file (train_source*.tsv or test_source*.tsv).

    Args:
        filepath: Absolute or relative path to the TSV.
        expected_prefix: Expected entity_id prefix ('S1-', 'S2-', 'S3-').
                         If given, logs a warning for non-matching IDs.

    Returns:
        DataFrame with at minimum columns: entity_id, business_name,
        business_address, country.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    df = _read_tsv(filepath)

    if "entity_id" not in df.columns:
        raise ValueError(f"Missing 'entity_id' column in {filepath}. Found: {df.columns.tolist()}")

    if expected_prefix:
        bad = (~df["entity_id"].str.startswith(expected_prefix)).sum()
        if bad:
            logger.warning(
                "%.0f IDs in %s do not start with '%s'", bad, filepath, expected_prefix
            )

    # Ensure canonical columns exist (fill with empty if absent)
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            logger.warning("Column '%s' missing in %s — filling with empty string.", col, filepath)
            df[col] = ""

    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load the training ground truth TSV.

    Expected columns: source1_entity_id, matched_entity_ids
    (where matched_entity_ids is a comma-separated list or empty).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth not found: {filepath}")
    return _read_tsv(filepath)


def load_train_data(
    train_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all four training files.

    Returns:
        (s1_df, s2_df, s3_df, gt_df)
    """
    d = Path(train_dir)
    s1 = load_source_file(str(d / "train_source1.tsv"), "S1-")
    s2 = load_source_file(str(d / "train_source2.tsv"), "S2-")
    s3 = load_source_file(str(d / "train_source3.tsv"), "S3-")
    gt = load_ground_truth(str(d / "train_ground_truth.tsv"))

    logger.info("Train loaded — S1: %d, S2: %d, S3: %d, GT: %d", len(s1), len(s2), len(s3), len(gt))
    return s1, s2, s3, gt


def load_test_data(
    test_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three test files.

    Returns:
        (s1_df, s2_df, s3_df)
    """
    d = Path(test_dir)
    s1 = load_source_file(str(d / "test_source1.tsv"), "S1-")
    s2 = load_source_file(str(d / "test_source2.tsv"), "S2-")
    s3 = load_source_file(str(d / "test_source3.tsv"), "S3-")

    logger.info("Test loaded — S1: %d, S2: %d, S3: %d", len(s1), len(s2), len(s3))
    return s1, s2, s3


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parse ground truth DataFrame into:
        { source1_entity_id -> [matched_entity_id, ...] }

    Handles:
    - Single match per row
    - Comma-separated list in matched_entity_ids column
    - Empty / singleton (no match) rows
    """
    # Auto-detect column names
    cols = gt_df.columns.tolist()
    s1_col = None
    match_col = None

    for c in cols:
        if "source1" in c or c in ("s1_id", "s1_entity_id", "source1_id"):
            s1_col = c
        if "match" in c:
            match_col = c

    # Fallback: first col = s1_id, second = matches
    if s1_col is None:
        s1_col = cols[0]
    if match_col is None:
        match_col = cols[1] if len(cols) > 1 else cols[0]

    result: Dict[str, List[str]] = {}
    for s1_val, match_val in zip(gt_df[s1_col], gt_df[match_col]):
        s1_id = str(s1_val).strip()
        raw = str(match_val).strip()

        if raw == "" or raw.lower() in ("nan", "none", "-"):
            result[s1_id] = []
        else:
            matches = [m.strip() for m in raw.split(",") if m.strip()]
            result[s1_id] = matches

    return result


# ---------------------------------------------------------------------------
# Data Audit
# ---------------------------------------------------------------------------

def audit_source(df: pd.DataFrame, name: str) -> Dict:
    """
    Compute a comprehensive data audit for a single source DataFrame.

    Returns a dict of audit statistics.
    """
    audit: Dict = {"source": name, "rows": len(df), "columns": df.columns.tolist()}

    # Entity ID integrity
    if "entity_id" in df.columns:
        audit["duplicate_ids"] = int(df["entity_id"].duplicated().sum())
        audit["unique_ids"] = int(df["entity_id"].nunique())

    # Null / empty analysis
    null_counts = df.isnull().sum().to_dict()
    empty_counts = {c: int((df[c] == "").sum()) for c in df.columns if df[c].dtype == object}
    audit["null_counts"] = null_counts
    audit["empty_string_counts"] = empty_counts

    # Country
    if "country" in df.columns:
        audit["country_value_counts"] = df["country"].value_counts().to_dict()
        audit["missing_country"] = int((df["country"] == "").sum())
        audit["unique_countries"] = int(df["country"].nunique())

    # Business name
    if "business_name" in df.columns:
        lengths = df["business_name"].str.len()
        audit["name_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
            "empty_count": int((df["business_name"] == "").sum()),
            "pct_empty": float((df["business_name"] == "").mean() * 100),
        }
        audit["duplicate_names"] = int(df["business_name"].duplicated(keep=False).sum())

    # Business address
    if "business_address" in df.columns:
        lengths = df["business_address"].str.len()
        audit["address_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
            "empty_count": int((df["business_address"] == "").sum()),
            "pct_empty": float((df["business_address"] == "").mean() * 100),
        }
        audit["duplicate_addresses"] = int(df["business_address"].duplicated(keep=False).sum())

    return audit


def audit_ground_truth(gt_dict: Dict[str, List[str]]) -> Dict:
    """Compute match statistics from the parsed ground truth dict."""
    match_counts = [len(v) for v in gt_dict.values()]
    n_s1 = len(match_counts)
    n_singletons = sum(1 for c in match_counts if c == 0)
    n_s2_matches = sum(1 for v in gt_dict.values() for mid in v if mid.startswith("S2-"))
    n_s3_matches = sum(1 for v in gt_dict.values() for mid in v if mid.startswith("S3-"))

    return {
        "total_s1_entities": n_s1,
        "total_matches": sum(match_counts),
        "singleton_count": n_singletons,
        "singleton_pct": float(n_singletons / n_s1 * 100) if n_s1 else 0.0,
        "match_count_mean": float(np.mean(match_counts)) if match_counts else 0.0,
        "match_count_median": float(np.median(match_counts)) if match_counts else 0.0,
        "match_count_max": int(max(match_counts)) if match_counts else 0,
        "s2_matches": n_s2_matches,
        "s3_matches": n_s3_matches,
        "distribution": {
            str(k): int(v)
            for k, v in sorted(
                pd.Series(match_counts).value_counts().to_dict().items()
            )
        },
    }


def run_full_audit(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Run full data audit and print a human-readable summary.

    Returns the combined audit dict.
    """
    audits = {
        "source1": audit_source(s1, "Source 1"),
        "source2": audit_source(s2, "Source 2"),
        "source3": audit_source(s3, "Source 3"),
    }
    if gt_dict is not None:
        audits["ground_truth"] = audit_ground_truth(gt_dict)

    _print_audit(audits)
    return audits


def _print_audit(audits: Dict):
    """Print a formatted audit summary."""
    print("\n" + "=" * 70)
    print("DATA AUDIT REPORT")
    print("=" * 70)

    for src in ["source1", "source2", "source3"]:
        a = audits.get(src, {})
        if not a:
            continue
        print(f"\n--- {a.get('source', src)} ---")
        print(f"  Rows        : {a.get('rows', '?')}")
        print(f"  Columns     : {a.get('columns', [])}")
        print(f"  Unique IDs  : {a.get('unique_ids', '?')}  |  Duplicate IDs: {a.get('duplicate_ids', '?')}")

        if "country_value_counts" in a:
            print(f"  Countries   : {a['country_value_counts']}")
            print(f"  Missing country: {a.get('missing_country', 0)}")

        if "name_length" in a:
            nl = a["name_length"]
            print(
                f"  Name length : mean={nl['mean']:.1f} median={nl['median']:.1f} "
                f"max={nl['max']}  empty={nl['empty_count']} ({nl['pct_empty']:.1f}%)"
            )

        if "address_length" in a:
            al = a["address_length"]
            print(
                f"  Addr length : mean={al['mean']:.1f} median={al['median']:.1f} "
                f"max={al['max']}  empty={al['empty_count']} ({al['pct_empty']:.1f}%)"
            )

        ec = a.get("empty_string_counts", {})
        if ec:
            print(f"  Empty counts: {ec}")

    if "ground_truth" in audits:
        g = audits["ground_truth"]
        print("\n--- Ground Truth ---")
        print(f"  S1 entities         : {g['total_s1_entities']}")
        print(f"  Total matches       : {g['total_matches']}")
        print(f"  Singletons (no match): {g['singleton_count']} ({g['singleton_pct']:.1f}%)")
        print(f"  Match count mean    : {g['match_count_mean']:.2f}")
        print(f"  Match count median  : {g['match_count_median']:.2f}")
        print(f"  Match count max     : {g['match_count_max']}")
        print(f"  S2 matches          : {g['s2_matches']}")
        print(f"  S3 matches          : {g['s3_matches']}")
        print(f"  Match distribution  : {g['distribution']}")

    print("\n" + "=" * 70)
