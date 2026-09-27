"""
submission.py — Output generation, internal validation, and submission versioning.

Generates:
  output/matching_results.tsv   — source1_entity_id <TAB> matched_entity_ids
  output/candidate_pairs.tsv    — source1_entity_id <TAB> candidate_entity_ids

Internal validation checks BEFORE the official validator:
- Every S1 test entity appears exactly once
- No duplicate S1 IDs
- No duplicate match IDs per row
- Matches ⊆ candidates
- Only S2/S3 IDs in match lists
- No NaN / "nan" / list syntax
- Correct tab-separated format
"""

import csv
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_matching_results(
    predictions: Dict[str, List[str]],
    s1_ids: List[str],
    output_path: str,
) -> str:
    """
    Write matching_results.tsv.

    Format: source1_entity_id <TAB> matched_entity_ids
    where matched_entity_ids is comma-separated (or empty).

    Every S1 ID appears exactly once. Order follows s1_ids list.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids:
            matches = predictions.get(s1_id, [])
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for mid in matches:
                if mid not in seen:
                    seen.add(mid)
                    deduped.append(mid)
            match_str = ",".join(deduped)
            writer.writerow([s1_id, match_str])

    logger.info("Wrote matching_results.tsv: %s (%d rows)", output_path, len(s1_ids))
    return output_path


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    s1_ids: List[str],
    output_path: str,
) -> str:
    """
    Write candidate_pairs.tsv.

    Format: source1_entity_id <TAB> candidate_entity_ids
    where candidate_entity_ids is comma-separated (or empty).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in s1_ids:
            cands = candidates.get(s1_id, set())
            # Ensure consistent ordering
            cand_list = sorted(cands)
            cand_str = ",".join(cand_list)
            writer.writerow([s1_id, cand_str])

    logger.info("Wrote candidate_pairs.tsv: %s (%d rows)", output_path, len(s1_ids))
    return output_path


def generate_outputs(
    predictions: Dict[str, List[str]],
    candidates: Dict[str, Set[str]],
    s1_ids: List[str],
    output_dir: str,
) -> tuple:
    """
    Generate both output files and run internal validation.

    Returns:
        (matching_path, candidate_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    write_matching_results(predictions, s1_ids, matching_path)
    write_candidate_pairs(candidates, s1_ids, candidate_path)

    return matching_path, candidate_path


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def validate_outputs(
    matching_path: str,
    candidate_path: str,
    expected_s1_ids: List[str],
    expected_s23_ids: Optional[Set[str]] = None,
) -> bool:
    """
    Run internal sanity checks on the output files.

    Checks:
    1. Files exist
    2. Correct tab-separated format
    3. Correct header columns
    4. Every expected S1 ID present exactly once
    5. No duplicate S1 IDs
    6. No duplicate match/candidate IDs per row
    7. Matches ⊆ candidates (for each S1)
    8. Only S2-/S3- prefixed IDs in match lists
    9. No "nan", "None", "[]" strings
    10. No extra S1 IDs not in expected list

    Returns:
        True if all checks pass.
    """
    print("\n--- Internal Output Validation ---")
    errors = []
    warnings = []

    # 1. Files exist
    for path in [matching_path, candidate_path]:
        if not os.path.exists(path):
            errors.append(f"File does not exist: {path}")

    if errors:
        _print_validation_result(errors, warnings)
        return False

    # 2. Load files
    try:
        matching_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
        candidate_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        errors.append(f"Failed to read output files: {e}")
        _print_validation_result(errors, warnings)
        return False

    # 3. Header columns
    expected_match_cols = ["source1_entity_id", "matched_entity_ids"]
    expected_cand_cols = ["source1_entity_id", "candidate_entity_ids"]

    if list(matching_df.columns) != expected_match_cols:
        errors.append(f"Matching columns wrong: {list(matching_df.columns)} != {expected_match_cols}")
    if list(candidate_df.columns) != expected_cand_cols:
        errors.append(f"Candidate columns wrong: {list(candidate_df.columns)} != {expected_cand_cols}")

    # 4-5. S1 ID coverage and duplicates
    expected_s1_set = set(expected_s1_ids)
    found_match_s1 = matching_df["source1_entity_id"].tolist()
    found_cand_s1 = candidate_df["source1_entity_id"].tolist()

    found_match_s1_set = set(found_match_s1)
    found_cand_s1_set = set(found_cand_s1)

    missing_in_match = expected_s1_set - found_match_s1_set
    missing_in_cand = expected_s1_set - found_cand_s1_set
    extra_in_match = found_match_s1_set - expected_s1_set
    extra_in_cand = found_cand_s1_set - expected_s1_set

    if missing_in_match:
        errors.append(f"matching_results.tsv missing {len(missing_in_match)} S1 IDs: {list(missing_in_match)[:5]}")
    if missing_in_cand:
        errors.append(f"candidate_pairs.tsv missing {len(missing_in_cand)} S1 IDs: {list(missing_in_cand)[:5]}")
    if extra_in_match:
        errors.append(f"matching_results.tsv has unexpected S1 IDs: {list(extra_in_match)[:5]}")
    if extra_in_cand:
        errors.append(f"candidate_pairs.tsv has unexpected S1 IDs: {list(extra_in_cand)[:5]}")

    # Duplicate S1 IDs
    dup_match = pd.Series(found_match_s1).duplicated().sum()
    dup_cand = pd.Series(found_cand_s1).duplicated().sum()
    if dup_match:
        errors.append(f"matching_results.tsv has {dup_match} duplicate S1 IDs")
    if dup_cand:
        errors.append(f"candidate_pairs.tsv has {dup_cand} duplicate S1 IDs")

    # 6-9. Per-row checks
    # Build match dict and candidate dict from files
    match_dict: Dict[str, List[str]] = {}
    cand_dict: Dict[str, List[str]] = {}

    bad_strings = {"nan", "none", "[]", "", "na"}

    for s1_id, raw_val in zip(matching_df["source1_entity_id"], matching_df["matched_entity_ids"]):
        raw = str(raw_val).strip()
        if raw == "":
            ids = []
        else:
            ids = [m.strip() for m in raw.split(",") if m.strip()]
        match_dict[s1_id] = ids

        for mid in ids:
            if mid.lower() in bad_strings:
                errors.append(f"Bad match ID '{mid}' in row for {s1_id}")
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                errors.append(f"Match ID '{mid}' not S2-/S3- prefix for {s1_id}")

        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate match IDs for {s1_id}: {ids}")

    for s1_id, raw_val in zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_ids"]):
        raw = str(raw_val).strip()
        if raw == "":
            ids = []
        else:
            ids = [m.strip() for m in raw.split(",") if m.strip()]
        cand_dict[s1_id] = ids

        if len(ids) != len(set(ids)):
            warnings.append(f"Duplicate candidate IDs for {s1_id}")

    # 7. Matches ⊆ candidates
    for s1_id in expected_s1_ids:
        m_set = set(match_dict.get(s1_id, []))
        c_set = set(cand_dict.get(s1_id, []))
        not_in_cands = m_set - c_set
        if not_in_cands:
            errors.append(
                f"Match IDs not in candidates for {s1_id}: {not_in_cands}"
            )

    # 8. S2/S3 IDs only (already checked per-row above for matches)
    # Check candidates too
    if expected_s23_ids is not None:
        for s1_id, cand_ids in cand_dict.items():
            for cid in cand_ids:
                if cid not in expected_s23_ids:
                    warnings.append(f"Candidate ID '{cid}' not in expected S23 pool for {s1_id}")

    _print_validation_result(errors, warnings)
    return len(errors) == 0


def _print_validation_result(errors: List[str], warnings: List[str]):
    if errors:
        print(f"  [FAIL] {len(errors)} errors:")
        for e in errors[:20]:
            print(f"    ERROR: {e}")
    else:
        print("  [PASS] All internal validation checks passed.")
    if warnings:
        print(f"  {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"    WARN: {w}")


# ---------------------------------------------------------------------------
# Submission versioning
# ---------------------------------------------------------------------------

def save_submission_version(
    matching_path: str,
    candidate_path: str,
    config: dict,
    metrics: dict,
    description: str,
    submissions_dir: str,
) -> str:
    """
    Archive a submission version with metadata.

    Creates a timestamped directory under submissions_dir containing:
    - matching_results.tsv
    - candidate_pairs.tsv
    - metadata.json

    Returns the version directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(submissions_dir, f"v_{timestamp}")
    os.makedirs(version_dir, exist_ok=True)

    shutil.copy2(matching_path, os.path.join(version_dir, "matching_results.tsv"))
    shutil.copy2(candidate_path, os.path.join(version_dir, "candidate_pairs.tsv"))

    metadata = {
        "timestamp": timestamp,
        "description": description,
        "metrics": metrics,
        "config": _sanitize_config(config),
    }
    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Submission saved to: %s", version_dir)
    return version_dir


def _sanitize_config(config: Any) -> Any:
    """Make config JSON-serializable."""
    if isinstance(config, dict):
        return {k: _sanitize_config(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_sanitize_config(x) for x in config]
    if isinstance(config, (int, float, str, bool, type(None))):
        return config
    return str(config)


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def log_experiment(
    experiment_id: str,
    metrics: dict,
    config_summary: dict,
    experiments_path: str,
):
    """
    Append an experiment record to experiments/results.csv.
    """
    import csv as csv_mod
    os.makedirs(os.path.dirname(experiments_path), exist_ok=True)

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "macro_f05": metrics.get("macro_f05", ""),
        "macro_precision": metrics.get("macro_precision", ""),
        "macro_recall": metrics.get("macro_recall", ""),
        "singleton_accuracy": metrics.get("singleton_accuracy", ""),
        "blocking_recall": metrics.get("blocking_recall", ""),
        "threshold": config_summary.get("threshold", ""),
        "model": config_summary.get("model", ""),
        "description": config_summary.get("description", ""),
    }

    file_exists = os.path.exists(experiments_path)
    with open(experiments_path, "a", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Experiment logged: %s → F0.5=%.4f", experiment_id, metrics.get("macro_f05", 0))
