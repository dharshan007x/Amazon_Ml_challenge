"""
test_blocking_benchmark.py — Empirical benchmark for candidate generation recall and latency.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalization import normalize_name, normalize_address, normalize_country

def run_benchmark():
    print("Loading ground truth sample...")
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=1000)
    s1_needed = set(gt_df["source1_entity_id"])
    true_matches_map = {}
    needed_s2 = set()
    needed_s3 = set()
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        m_str = str(r["matched_entity_ids"])
        if pd.notna(m_str) and m_str.strip():
            mids = [x.strip() for x in m_str.split(",") if x.strip()]
            true_matches_map[s1_id] = mids
            for mid in mids:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

    print(f"Sampling S1 records ({len(s1_needed)} needed)...")
    s1_data = []
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=200000):
        match = chunk[chunk["entity_id"].isin(s1_needed)]
        s1_data.append(match)
        if sum(len(x) for x in s1_data) == len(s1_needed):
            break
    s1_df = pd.concat(s1_data, ignore_index=True)

    print(f"Sampling true matches from S2 ({len(needed_s2)}) and S3 ({len(needed_s3)})...")
    s23_data = []
    for s_file, needed in [
        ("dataset/train/train_source2.tsv", needed_s2),
        ("dataset/train/train_source3.tsv", needed_s3),
    ]:
        for chunk in pd.read_csv(s_file, sep="\t", chunksize=250000):
            match = chunk[chunk["entity_id"].isin(needed)]
            s23_data.append(match)

    # Add 30k random background records to test realistic multi-candidate ranking
    print("Adding 30,000 background records to candidate pool...")
    s23_bg = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=30000)
    s23_data.append(s23_bg)
    s23_df = pd.concat(s23_data, ignore_index=True).drop_duplicates("entity_id")

    print(f"Pool size: S1 = {len(s1_df)}, S23 = {len(s23_df)}")

    def prep_records(df):
        records = []
        eids = df["entity_id"].tolist()
        names = df["business_name"].tolist()
        addrs = df["business_address"].tolist()
        countries = df["country"].tolist()
        for eid, n, a, c in zip(eids, names, addrs, countries):
            nv = normalize_name(n)
            av = normalize_address(a)
            records.append({
                "entity_id": eid,
                "country": normalize_country(c),
                "name_clean": nv["cleaned"],
                "name_strip": nv["stripped"],
                "name_sort": nv["sorted_tokens"],
                "name_alnum": nv["alphanumeric"],
                "name_tokens": nv["tokens"].split(),
                "addr_clean": av["cleaned"],
                "addr_tokens": av["tokens"].split(),
                "postal": av["postal_candidates"][0] if av["postal_candidates"] else "",
                "first_num": av["first_numeric"]
            })
        return records

    t0 = time.time()
    s1_recs = prep_records(s1_df)
    s23_recs = prep_records(s23_df)
    print(f"Records normalized in {time.time()-t0:.2f}s")

    # Inverted index with bucket cap
    t0 = time.time()
    inv_index = defaultdict(list)
    token_doc_freq = defaultdict(int)

    for r in s23_recs:
        c = r["country"]
        for t in set(r["name_tokens"] + r["addr_tokens"]):
            token_doc_freq[(c, t)] += 1

    MAX_BUCKET = 120
    for r in s23_recs:
        eid = r["entity_id"]
        c = r["country"]
        if r["name_clean"]:
            inv_index[(c, "nc", r["name_clean"])].append(eid)
        if r["name_strip"]:
            inv_index[(c, "ns", r["name_strip"])].append(eid)
        if r["name_sort"]:
            inv_index[(c, "nso", r["name_sort"])].append(eid)
        if r["name_alnum"] and len(r["name_alnum"]) >= 4:
            inv_index[(c, "na", r["name_alnum"])].append(eid)
        if r["postal"]:
            inv_index[(c, "pos", r["postal"])].append(eid)
        if r["first_num"] and r["name_tokens"]:
            inv_index[(c, "num_tok", r["first_num"], r["name_tokens"][0])].append(eid)
        for tok in r["name_tokens"]:
            if len(tok) >= 3 and token_doc_freq[(c, tok)] <= MAX_BUCKET:
                inv_index[(c, "ntok", tok)].append(eid)
        for tok in r["addr_tokens"]:
            if len(tok) >= 3 and token_doc_freq[(c, tok)] <= MAX_BUCKET:
                inv_index[(c, "atok", tok)].append(eid)

    print(f"Inverted index built in {time.time()-t0:.2f}s (distinct keys: {len(inv_index):,})")

    # Candidate retrieval
    t0 = time.time()
    candidates = {}
    for r in s1_recs:
        eid = r["entity_id"]
        c = r["country"]
        scores = defaultdict(int)

        # High confidence exact matches
        if r["name_clean"]:
            for cid in inv_index.get((c, "nc", r["name_clean"]), []):
                scores[cid] += 8
        if r["name_strip"]:
            for cid in inv_index.get((c, "ns", r["name_strip"]), []):
                scores[cid] += 6
        if r["name_sort"]:
            for cid in inv_index.get((c, "nso", r["name_sort"]), []):
                scores[cid] += 5
        if r["name_alnum"] and len(r["name_alnum"]) >= 4:
            for cid in inv_index.get((c, "na", r["name_alnum"]), []):
                scores[cid] += 5
        if r["postal"]:
            for cid in inv_index.get((c, "pos", r["postal"]), []):
                scores[cid] += 2
        if r["first_num"] and r["name_tokens"]:
            for cid in inv_index.get((c, "num_tok", r["first_num"], r["name_tokens"][0]), []):
                scores[cid] += 3

        # Token overlap
        for tok in r["name_tokens"]:
            if len(tok) >= 3 and token_doc_freq[(c, tok)] <= MAX_BUCKET:
                for cid in inv_index.get((c, "ntok", tok), []):
                    scores[cid] += 2
        for tok in r["addr_tokens"]:
            if len(tok) >= 3 and token_doc_freq[(c, tok)] <= MAX_BUCKET:
                for cid in inv_index.get((c, "atok", tok), []):
                    scores[cid] += 1

        # Keep top 35 candidates
        top_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]
        candidates[eid] = {cid for cid, _ in top_cands}

    dt = time.time() - t0
    print(f"Candidates generated in {dt:.2f}s ({len(s1_recs)/dt:.0f} queries/sec)")

    # Evaluate blocking recall
    total_true = 0
    found_true = 0
    for s1_id, true_mids in true_matches_map.items():
        c_set = candidates.get(s1_id, set())
        for mid in true_mids:
            if mid in needed_s2 or mid in needed_s3:
                total_true += 1
                if mid in c_set:
                    found_true += 1

    recall = found_true / max(total_true, 1)
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    print("=" * 60)
    print(f"BLOCKING RECALL: {found_true}/{total_true} ({recall*100:.2f}%)")
    print(f"AVERAGE CANDIDATES PER S1: {avg_cands:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
