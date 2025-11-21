#!/usr/bin/env python3
# scripts/sample_for_labeling_stream.py
"""
Memory-efficient stratified sampling for labeling.

Writes:
 - data/processed/sampled_pos.parquet
 - data/processed/sampled_neg.parquet

Strategy:
 - Build a signature set for candidate rows (hash(title+channelTitle+publish_time))
 - Reservoir-sample per-country quotas from candidates (streaming)
 - Stream ALL and reservoir-sample negatives per-country, skipping candidate signatures.
"""
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import argparse, os, hashlib, random, json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", default="data/processed/candidates.parquet")
parser.add_argument("--all", default="data/processed/ALL_youtube_trending_data.parquet")
parser.add_argument("--out_pos", default="data/processed/sampled_pos.parquet")
parser.add_argument("--out_neg", default="data/processed/sampled_neg.parquet")
parser.add_argument("--pos_per_country", type=int, default=500)
parser.add_argument("--neg_per_country", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=2000)
args = parser.parse_args()

random.seed(args.seed)

def sig_for_row(row):
    # row is dict-like with keys possibly missing
    # combine stable fields to create a signature
    t = (row.get("title") or "") + "|" + (str(row.get("channelTitle") or "")) + "|" + (str(row.get("publish_time") or ""))
    return hashlib.sha1(t.encode("utf8")).hexdigest()

def reservoir_add(reservoir, k, item):
    # reservoir is list; item appended with probability if >k
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        # replace with probability k/(n_seen)
        # we track n_seen externally by passing a counter (we'll maintain counters separate)
        raise RuntimeError("Use reservoir_add_with_count instead")

def reservoir_add_with_count(reservoir, k, item, count):
    # count is 1-based count of items seen for this key
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        # rand int in [0, count-1]
        r = random.randrange(0, count)
        if r < k:
            reservoir[r] = item

def stream_signatures_and_sample_pos():
    # First pass: stream candidates, build sig set and per-country reservoirs for positives
    cand_ds = ds.dataset(args.candidates, format="parquet")
    scanner = cand_ds.scanner(batch_size=args.batch_size)
    sigset = set()
    pos_res = defaultdict(list)        # country -> list of row dicts
    pos_counts = defaultdict(int)      # country -> seen count for reservoir
    total_seen = 0
    for batch in scanner.to_batches():
        table = batch
        # convert only needed columns to python lists
        cols = table.column_names
        # convert batch to pandas just for simplicity on small batch sizes
        df = table.to_pandas()
        for idx, row in df.iterrows():
            total_seen += 1
            d = row.to_dict()
            s = sig_for_row(d)
            sigset.add(s)
            country = d.get("country_code") or "XX"
            pos_counts[country] += 1
            # reservoir sampling per country
            reservoir_add_with_count(pos_res[country], args.pos_per_country, d, pos_counts[country])
    return sigset, pos_res

def stream_sample_neg(sigset):
    # Stream all rows and sample negatives per country skipping candidate signatures
    all_ds = ds.dataset(args.all, format="parquet")
    scanner = all_ds.scanner(batch_size=args.batch_size)
    neg_res = defaultdict(list)
    neg_counts = defaultdict(int)
    total_seen = 0
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        for idx, row in df.iterrows():
            total_seen += 1
            d = row.to_dict()
            s = sig_for_row(d)
            if s in sigset:
                continue
            country = d.get("country_code") or "XX"
            neg_counts[country] += 1
            reservoir_add_with_count(neg_res[country], args.neg_per_country, d, neg_counts[country])
    return neg_res

def flatten_reservoirs(res_dict):
    rows = []
    for country, lst in res_dict.items():
        rows.extend(lst)
    return rows

def save_parquet(rows, outpath):
    if not rows:
        print("No rows to save for", outpath)
        return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    df.to_parquet(outpath, index=False)
    print("Wrote", len(df), "rows to", outpath)

def main():
    print("Stage 1: streaming candidates to build signature set and sample positives")
    sigset, pos_res = stream_signatures_and_sample_pos()
    print("Candidates signatures:", len(sigset))
    pos_rows = flatten_reservoirs(pos_res)
    print("Pos samples collected (approx):", len(pos_rows))

    print("Stage 2: streaming ALL to sample negatives (skipping candidate signatures)")
    neg_res = stream_sample_neg(sigset)
    neg_rows = flatten_reservoirs(neg_res)
    print("Neg samples collected (approx):", len(neg_rows))

    print("Saving outputs...")
    save_parquet(pos_rows, args.out_pos)
    save_parquet(neg_rows, args.out_neg)
    print("Done.")

if __name__ == "__main__":
    main()
