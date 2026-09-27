"""
main.py — Master orchestration script for the Amazon ML Challenge pipeline.

Phases:
  Phase 1: Data loading and audit
  Phase 2: Normalization
  Phase 3: Blocking (candidate generation)
  Phase 4: Feature engineering
  Phase 5: Training pair construction
  Phase 6: Baseline model training
  Phase 7: Hard negative mining + retraining
  Phase 8: Threshold optimization (macro F0.5)
  Phase 9: Test inference (country-partitioned, memory-safe)
  Phase 10: Output generation and internal validation
  Phase 11: Official submission validation & zip packaging

Usage:
  python code/business_entity_resolution/src/main.py                          # run full pipeline
  python code/business_entity_resolution/src/main.py --config configs/default.yaml
  python code/business_entity_resolution/src/main.py --phase train            # only train
  python code/business_entity_resolution/src/main.py --phase infer            # only inference
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_train_data, load_test_data, parse_ground_truth, run_full_audit
)
from src.normalization import normalize_dataframe, normalize_country
from src.blocking import BlockingEngine
from src.features import FeatureEngine
from src.training import (
    build_training_pairs, build_feature_matrix,
    mine_hard_negatives, train_model, optimize_threshold,
    save_model, load_model
)
from src.evaluation import evaluate, print_evaluation, evaluate_blocking, error_analysis
from src.decision import DecisionEngine
from src.submission import (
    generate_outputs, validate_outputs,
    save_submission_version, log_experiment
)

logger = logging.getLogger("er_pipeline")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def run_pipeline(config_path: str = "code/business_entity_resolution/configs/default.yaml", phase: str = "all"):
    """
    Execute the entity resolution challenge pipeline.
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logger.info("Loading config: %s", config_path)
    cfg = load_config(config_path)
    seed = cfg.get("random_seed", 42)
    set_random_seeds(seed)

    train_dir = cfg["data"]["train_dir"]
    test_dir = cfg["data"]["test_dir"]
    output_dir = cfg["data"]["output_dir"]

    model_path = os.path.join(cfg["paths"]["models"], "model.pkl")
    feature_engine_path = os.path.join(cfg["paths"]["models"], "feature_engine.pkl")
    meta_path = os.path.join(cfg["paths"]["models"], "pipeline_meta.json")

    best_threshold = cfg.get("decision", {}).get("default_threshold", 0.75)
    best_s2_thr = best_threshold
    best_s3_thr = best_threshold

    # ======================================================================
    # TRAINING STAGE (Phases 1-8)
    # ======================================================================
    if phase in ("all", "train"):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: MODEL TRAINING & THRESHOLD OPTIMIZATION")
        logger.info("=" * 60)

        # 1. Load ground truth
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        train_s1_path = os.path.join(train_dir, "train_source1.tsv")
        train_s2_path = os.path.join(train_dir, "train_source2.tsv")
        train_s3_path = os.path.join(train_dir, "train_source3.tsv")

        logger.info("Reading ground truth from %s...", gt_path)
        train_gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        gt_dict = parse_ground_truth(train_gt_df)

        # Sample stratified S1 entities for training to optimize memory and speed
        train_sample_size = cfg.get("training", {}).get("train_sample_size", 40000)
        all_s1_gt_ids = list(gt_dict.keys())
        rng = random.Random(seed)
        if train_sample_size and train_sample_size < len(all_s1_gt_ids):
            sampled_s1_ids = set(rng.sample(all_s1_gt_ids, train_sample_size))
            logger.info("Using stratified training sample: %d / %d S1 entities", len(sampled_s1_ids), len(all_s1_gt_ids))
        else:
            sampled_s1_ids = set(all_s1_gt_ids)

        # Collect required S2 and S3 true matches
        needed_s2 = set()
        needed_s3 = set()
        for sid in sampled_s1_ids:
            for mid in gt_dict.get(sid, []):
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

        # Load S1 records for sample
        logger.info("Loading S1 records for sample...")
        s1_chunks = []
        for chunk in pd.read_csv(train_s1_path, sep="\t", dtype=str, chunksize=200000):
            m = chunk[chunk["entity_id"].isin(sampled_s1_ids)]
            if len(m) > 0:
                s1_chunks.append(m)
            if sum(len(x) for x in s1_chunks) == len(sampled_s1_ids):
                break
        train_s1 = pd.concat(s1_chunks, ignore_index=True)

        # Load S2 and S3 matches + random background records
        logger.info("Loading S2 records (%d matches + background)...", len(needed_s2))
        s2_chunks = []
        for chunk in pd.read_csv(train_s2_path, sep="\t", dtype=str, chunksize=250000):
            m = chunk[chunk["entity_id"].isin(needed_s2)]
            if len(m) > 0:
                s2_chunks.append(m)
        # Background
        s2_bg = pd.read_csv(train_s2_path, sep="\t", dtype=str, nrows=50000)
        s2_chunks.append(s2_bg)
        train_s2 = pd.concat(s2_chunks, ignore_index=True).drop_duplicates("entity_id")

        logger.info("Loading S3 records (%d matches + background)...", len(needed_s3))
        s3_chunks = []
        for chunk in pd.read_csv(train_s3_path, sep="\t", dtype=str, chunksize=250000):
            m = chunk[chunk["entity_id"].isin(needed_s3)]
            if len(m) > 0:
                s3_chunks.append(m)
        s3_bg = pd.read_csv(train_s3_path, sep="\t", dtype=str, nrows=50000)
        s3_chunks.append(s3_bg)
        train_s3 = pd.concat(s3_chunks, ignore_index=True).drop_duplicates("entity_id")

        logger.info("Normalizing training records...")
        train_s1 = normalize_dataframe(train_s1)
        train_s2 = normalize_dataframe(train_s2)
        train_s3 = normalize_dataframe(train_s3)
        train_s23 = pd.concat([train_s2, train_s3], ignore_index=True).drop_duplicates("entity_id")

        # Blocking on training set
        logger.info("Fitting BlockingEngine on training pool...")
        train_blocking = BlockingEngine(cfg.get("blocking", {}))
        train_blocking.fit(train_s2, train_s3)
        train_candidates = train_blocking.get_candidates(train_s1)

        b_eval = train_blocking.evaluate_blocking(train_candidates, gt_dict, train_s1["entity_id"].tolist())
        logger.info("Train Blocking Recall: %.4f (Avg cands: %.1f)", b_eval["blocking_recall"], b_eval["avg_candidates_per_s1"])

        # Feature Engine
        logger.info("Fitting FeatureEngine on training corpus...")
        feature_engine = FeatureEngine(cfg.get("features", {}))
        feature_engine.fit([train_s1, train_s2, train_s3])

        # Pair construction
        logger.info("Constructing training pairs...")
        train_cfg = cfg.get("training", {})
        neg_ratio = train_cfg.get("random_negative_ratio", 6)
        pos_pairs, neg_pairs = build_training_pairs(
            train_s1, train_s23, gt_dict, train_candidates,
            negative_ratio=neg_ratio, random_seed=seed
        )

        all_pairs = pos_pairs + neg_pairs
        all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        logger.info("Total pairs: %d (%d pos, %d neg)", len(all_pairs), len(pos_pairs), len(neg_pairs))

        # Build feature matrix
        train_s1_lookup = train_s1.set_index("entity_id").to_dict("index")
        train_s23_lookup = train_s23.set_index("entity_id").to_dict("index")

        X_all, y_all, feat_names = build_feature_matrix(
            all_pairs, all_labels, feature_engine, train_s1_lookup, train_s23_lookup
        )
        logger.info("Feature matrix shape: %s", str(X_all.shape))

        # Train / Validation Split (Entity-level)
        val_frac = train_cfg.get("val_fraction", 0.2)
        unique_s1 = list(set(p[0] for p in all_pairs))
        rng.shuffle(unique_s1)
        n_val = max(1, int(len(unique_s1) * val_frac))
        val_s1_set = set(unique_s1[:n_val])
        train_s1_set = set(unique_s1[n_val:])

        train_mask = np.array([p[0] in train_s1_set for p in all_pairs])
        val_mask = ~train_mask

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_va, y_va = X_all[val_mask], y_all[val_mask]
        val_pairs = [all_pairs[i] for i in range(len(all_pairs)) if val_mask[i]]

        # Train baseline model
        logger.info("Training LightGBM model...")
        model_cfg = cfg.get("model", {})
        current_model = train_model(X_tr, y_tr, X_va, y_va, model_cfg)

        # Hard negative mining (1 iteration)
        logger.info("Mining hard negatives...")
        hard_negs = mine_hard_negatives(
            model=current_model,
            pos_pairs=pos_pairs,
            all_candidates=train_candidates,
            gt_dict=gt_dict,
            feature_engine=feature_engine,
            s1_lookup=train_s1_lookup,
            s23_lookup=train_s23_lookup,
            hard_negative_ratio=train_cfg.get("hard_negative_ratio", 4),
            score_threshold=0.30,
            random_seed=seed,
        )

        if hard_negs:
            logger.info("Retraining model with %d hard negatives...", len(hard_negs))
            aug_pairs = all_pairs + hard_negs
            aug_labels = all_labels + [0] * len(hard_negs)

            X_aug, y_aug, _ = build_feature_matrix(
                aug_pairs, aug_labels, feature_engine, train_s1_lookup, train_s23_lookup
            )
            aug_tr_mask = np.array([p[0] in train_s1_set for p in aug_pairs])
            aug_va_mask = ~aug_tr_mask

            current_model = train_model(
                X_aug[aug_tr_mask], y_aug[aug_tr_mask],
                X_aug[aug_va_mask], y_aug[aug_va_mask],
                model_cfg
            )
            final_val_pairs = [aug_pairs[i] for i in range(len(aug_pairs)) if aug_va_mask[i]]
            X_val_eval = X_aug[aug_va_mask]
        else:
            final_val_pairs = val_pairs
            X_val_eval = X_va

        # Threshold optimization for Macro F0.5
        logger.info("Optimizing threshold for Macro F0.5...")
        val_scores_raw = current_model.predict_proba(X_val_eval)[:, 1]
        val_score_map: Dict[str, Dict[str, float]] = {}
        for (s1_id, s23_id), sc in zip(final_val_pairs, val_scores_raw):
            if s1_id not in val_score_map:
                val_score_map[s1_id] = {}
            val_score_map[s1_id][s23_id] = float(sc)

        val_s1_list = sorted(val_s1_set)
        for sid in val_s1_list:
            if sid not in val_score_map:
                val_score_map[sid] = {}

        val_gt_local = {sid: gt_dict.get(sid, []) for sid in val_s1_list}
        threshold_cfg = cfg.get("decision", {})
        thresholds = threshold_cfg.get("threshold_sweep", [0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95])

        thr_opt = optimize_threshold(
            val_score_map, val_gt_local, val_s1_list,
            thresholds=thresholds,
            source_specific=threshold_cfg.get("source_specific_thresholds", True),
        )

        best_threshold = thr_opt["best_threshold"]
        best_s2_thr = thr_opt.get("best_s2_threshold", best_threshold)
        best_s3_thr = thr_opt.get("best_s3_threshold", best_threshold)
        logger.info("Optimal threshold: %.3f (Val Macro F0.5 = %.4f)", best_threshold, thr_opt["best_f05"])

        # Final evaluation on validation set
        val_preds = {
            sid: [mid for mid, sc in val_score_map.get(sid, {}).items() if sc >= best_threshold]
            for sid in val_s1_list
        }
        final_val_eval = evaluate(val_preds, val_gt_local, val_s1_list)
        print_evaluation(final_val_eval, f"Validation Evaluation (Threshold={best_threshold})")

        # Save model artifacts
        logger.info("Saving trained model to %s...", model_path)
        save_model(current_model, model_path)
        save_model(feature_engine, feature_engine_path)

        meta = {
            "best_threshold": best_threshold,
            "best_s2_threshold": best_s2_thr,
            "best_s3_threshold": best_s3_thr,
            "val_f05": final_val_eval["macro_f05"],
            "val_precision": final_val_eval["macro_precision"],
            "val_recall": final_val_eval["macro_recall"],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Clean up training data from memory
        del train_s1, train_s2, train_s3, train_s23, X_all, y_all
        gc.collect()

    # ======================================================================
    # TEST INFERENCE STAGE (Phases 9-11)
    # ======================================================================
    if phase in ("all", "infer"):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: TEST INFERENCE & OFFICIAL SUBMISSION GENERATION")
        logger.info("=" * 60)

        # Load model and feature engine
        logger.info("Loading model from %s...", model_path)
        final_model = load_model(model_path)
        feature_engine = load_model(feature_engine_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                best_threshold = meta.get("best_threshold", 0.75)
                best_s2_thr = meta.get("best_s2_threshold", best_threshold)
                best_s3_thr = meta.get("best_s3_threshold", best_threshold)

        logger.info("Using thresholds: default=%.3f, s2=%.3f, s3=%.3f", best_threshold, best_s2_thr, best_s3_thr)

        # Load ordered list of Test S1 entities
        test_s1_file = os.path.join(test_dir, "test_source1.tsv")
        test_s2_file = os.path.join(test_dir, "test_source2.tsv")
        test_s3_file = os.path.join(test_dir, "test_source3.tsv")

        logger.info("Reading test S1 entity IDs...")
        test_s1_full = pd.read_csv(test_s1_file, sep="\t", dtype=str).fillna("")
        ordered_test_s1_ids = test_s1_full["entity_id"].tolist()
        logger.info("Total test S1 entities: %d", len(ordered_test_s1_ids))

        matching_path = cfg["paths"]["matching_results"]
        candidate_path = cfg["paths"]["candidate_pairs"]
        os.makedirs(os.path.dirname(matching_path), exist_ok=True)

        final_predictions: Dict[str, List[str]] = {}
        final_candidates: Dict[str, Set[str]] = {}

        test_countries = [c for c in test_s1_full["country"].dropna().unique() if c]
        logger.info("Test countries to process: %s", test_countries)

        # S1 chunk size for blocking — keeps memory bounded even for large countries
        S1_CHUNK = 50_000
        # Feature scoring batch size
        SCORE_BATCH = 5_000

        for country in test_countries:
            logger.info("\n--- Processing Country: %s ---", country)

            # ── Load + normalize S2 and S3 for this country ─────────────────
            logger.info("  Loading S2 for %s...", country)
            c_s2_chunks = []
            for chunk in pd.read_csv(test_s2_file, sep="\t", dtype=str, chunksize=300_000):
                m = chunk[chunk["country"].str.lower() == country.lower()]
                if len(m) > 0:
                    c_s2_chunks.append(m)
            c_s2 = (pd.concat(c_s2_chunks, ignore_index=True)
                    if c_s2_chunks
                    else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]))
            c_s2 = normalize_dataframe(c_s2, compute_views=False)
            logger.info("  %s S2 records: %d", country, len(c_s2))

            logger.info("  Loading S3 for %s...", country)
            c_s3_chunks = []
            for chunk in pd.read_csv(test_s3_file, sep="\t", dtype=str, chunksize=300_000):
                m = chunk[chunk["country"].str.lower() == country.lower()]
                if len(m) > 0:
                    c_s3_chunks.append(m)
            c_s3 = (pd.concat(c_s3_chunks, ignore_index=True)
                    if c_s3_chunks
                    else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]))
            c_s3 = normalize_dataframe(c_s3, compute_views=False)
            logger.info("  %s S3 records: %d", country, len(c_s3))

            # Combined S23 for feature lookups — indexed on entity_id
            c_s23 = pd.concat([c_s2, c_s3], ignore_index=True).drop_duplicates("entity_id")
            c_s23_idx = c_s23.set_index("entity_id")  # used for feature lookup via .loc
            c_s23_id_set = set(c_s23["entity_id"])
            logger.info("  %s S23 combined: %d unique records", country, len(c_s23))

            # ── Fit BlockingEngine on S2+S3 (no giant dict) ─────────────────
            logger.info("  Fitting BlockingEngine for %s...", country)
            country_blocking = BlockingEngine(cfg.get("blocking", {}))
            country_blocking.fit(c_s2, c_s3)
            # Free S2/S3 raw DFs; blocking engine holds only lightweight index arrays
            del c_s2, c_s3
            gc.collect()
            logger.info("  BlockingEngine fit complete.")

            # ── Filter S1 for this country ───────────────────────────────────
            c_s1_full = test_s1_full[
                test_s1_full["country"].str.lower() == country.lower()
            ].copy()
            c_s1_full = normalize_dataframe(c_s1_full, compute_views=False)
            c_s1_ids_all = c_s1_full["entity_id"].tolist()
            logger.info("  %s S1 records: %d", country, len(c_s1_ids_all))

            # ── Process S1 in chunks to bound memory ─────────────────────────
            total_matched = 0
            for chunk_start in range(0, len(c_s1_ids_all), S1_CHUNK):
                chunk_ids = c_s1_ids_all[chunk_start: chunk_start + S1_CHUNK]
                c_s1_chunk = c_s1_full[c_s1_full["entity_id"].isin(set(chunk_ids))].copy()

                # Blocking candidates for this S1 chunk
                c_candidates = country_blocking.get_candidates(c_s1_chunk)

                # Build pairs, but only for candidates that exist in S23
                pairs_to_score = []
                for s1_id, cands in c_candidates.items():
                    valid_cands = cands & c_s23_id_set
                    final_candidates[s1_id] = valid_cands
                    for cid in valid_cands:
                        pairs_to_score.append((s1_id, cid))

                logger.info(
                    "  %s chunk %d-%d: %d pairs to score",
                    country, chunk_start, chunk_start + len(chunk_ids), len(pairs_to_score)
                )

                # Build lightweight lookup dicts ONLY for entities in this chunk
                chunk_s1_ids_set = set(chunk_ids)
                needed_s23_ids = {cid for _, cid in pairs_to_score}

                c_s1_lookup = c_s1_chunk.set_index("entity_id").to_dict("index")
                # Subset S23 index to only needed candidates — avoids full 3.8M dict
                c_s23_lookup = (
                    c_s23_idx.loc[c_s23_idx.index.isin(needed_s23_ids)]
                    .to_dict("index")
                )

                # Score pairs in batches
                for start in range(0, len(pairs_to_score), SCORE_BATCH):
                    batch = pairs_to_score[start: start + SCORE_BATCH]
                    X_batch, _ = feature_engine.compute_batch(batch, c_s1_lookup, c_s23_lookup)
                    if X_batch.shape[0] == 0:
                        continue
                    scores = final_model.predict_proba(X_batch)[:, 1]

                    for (s1_id, cid), sc in zip(batch, scores):
                        thr = best_s2_thr if cid.startswith("S2-") else best_s3_thr
                        if sc >= thr:
                            if s1_id not in final_predictions:
                                final_predictions[s1_id] = []
                            final_predictions[s1_id].append(cid)

                # Ensure every S1 in chunk has a prediction entry
                for s1_id in chunk_ids:
                    if s1_id not in final_predictions:
                        final_predictions[s1_id] = []

                chunk_matched = sum(1 for sid in chunk_ids if final_predictions.get(sid))
                total_matched += chunk_matched

                del c_s1_chunk, c_s1_lookup, c_s23_lookup, pairs_to_score
                gc.collect()

            logger.info(
                "  %s complete: %d / %d S1 entities matched",
                country, total_matched, len(c_s1_ids_all)
            )
            del c_s23, c_s23_idx, c_s1_full, country_blocking
            gc.collect()

        # Step 10: Generate output files
        logger.info("\nWriting output files...")
        matching_path, candidate_path = generate_outputs(
            final_predictions, final_candidates, ordered_test_s1_ids, output_dir
        )

        # Internal validation
        logger.info("Running internal validation...")
        valid = validate_outputs(
            matching_path, candidate_path, ordered_test_s1_ids
        )
        if not valid:
            logger.error("Internal validation FAILED! Please review issues above.")
            sys.exit(1)
        logger.info("Internal validation PASSED.")

        # Step 11: Submission Packaging
        sub_dir = save_submission_version(
            matching_path, candidate_path,
            cfg, meta if 'meta' in dir() else {},
            description="Country-partitioned multi-channel blocking + LightGBM + hard negative mining",
            submissions_dir=cfg["paths"]["submissions"],
        )

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info("Matching results : %s", matching_path)
        logger.info("Candidate pairs  : %s", candidate_path)
        logger.info("Submission saved : %s", sub_dir)

        # Official validator check
        validator_script = "utils/validate_submission.py"
        if os.path.exists(validator_script):
            logger.info("\nRunning official submission validator:")
            cmd = f'python {validator_script} --matching {matching_path} --candidate {candidate_path} --test-dir {test_dir}'
            logger.info("Command: %s", cmd)
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge — Entity Resolution Pipeline")
    parser.add_argument("--config", default="code/business_entity_resolution/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--phase", choices=["all", "train", "infer"], default="all", help="Phase to run: all, train, infer")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, phase=args.phase)
