#!/usr/bin/env python3
# scripts/add_country_by_join.py
"""
Build mapping from master ALL parquet and add country_code to candidates by lookup.
Streaming, memory-conscious.
"""
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import argparse, os, hashlib, json
from collections import defaultdict
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--all", default="data/processed/ALL_youtube_trending_data.parquet")
parser.add_argument("--candidates", default="data/processed/candidates.parquet")
parser.add_argument("--out", default="data/processed/candidates_with_country.parquet")
parser.add_argument("--batch_size", type=int, default=5000)
parser.add_argument("--use_signature_if_missing", action="store_true", help="Fall back to title|channel|publish_time signature if video_id missing")
args = parser.parse_args()

LOG = "logs/add_country_by_join_progress.txt"
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
open(LOG, "a").close()
def log(msg):
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")

def sig_from_row_dict(d):
    # stable fallback key if video_id missing
    t = (d.get("title") or "") + "|" + (d.get("channelTitle") or "") + "|" + (str(d.get("publish_time") or ""))
    return hashlib.sha1(t.encode("utf8")).hexdigest()

def build_map_from_all():
    log("Building mapping from ALL parquet: " + args.all)
    ds_all = ds.dataset(args.all, format="parquet")
    want_cols = []
    schema_names = ds_all.schema.names
    has_vid = "video_id" in schema_names
    has_country = "country_code" in schema_names
    if has_vid:
        want_cols = ["video_id","country_code"] if has_country else ["video_id"]
    else:
        # fallback to signature fields
        for c in ("title","channelTitle","publish_time","country_code"):
            if c in schema_names and c not in want_cols:
                want_cols.append(c)
    log(f"Schema has video_id={has_vid}, country_code={has_country}, using columns={want_cols}")

    mapping = dict()   # key -> country_code (first-seen)
    scanner = ds_all.scanner(columns=want_cols or None, batch_size=args.batch_size)
    seen = 0
    for batch in scanner.to_batches():
        # convert minimal columns to py lists
        cols = batch.column_names
        col_arrays = [batch.column(i) for i in range(batch.num_columns)]
        n = batch.num_rows
        for i in range(n):
            seen += 1
            try:
                if has_vid:
                    vid = batch.column(batch.schema.get_field_index("video_id"))[i].as_py()
                    c = batch.column(batch.schema.get_field_index("country_code"))[i].as_py() if "country_code" in batch.schema.names else None
                    if vid is None:
                        continue
                    if vid not in mapping and c is not None:
                        mapping[vid] = c
                else:
                    # build signature
                    title = batch.column(batch.schema.get_field_index("title"))[i].as_py() if "title" in batch.schema.names else ""
                    channel = batch.column(batch.schema.get_field_index("channelTitle"))[i].as_py() if "channelTitle" in batch.schema.names else ""
                    pub = batch.column(batch.schema.get_field_index("publish_time"))[i].as_py() if "publish_time" in batch.schema.names else ""
                    c = batch.column(batch.schema.get_field_index("country_code"))[i].as_py() if "country_code" in batch.schema.names else None
                    if c is None:
                        continue
                    s = hashlib.sha1(((title or "") + "|" + (channel or "") + "|" + (str(pub) or "")).encode("utf8")).hexdigest()
                    if s not in mapping:
                        mapping[s] = c
            except Exception:
                continue
        if seen % (args.batch_size * 10) == 0:
            log(f"scanned {seen} rows, mapping size={len(mapping)}")
    log(f"Finished building mapping: total scanned={seen}, mapping size={len(mapping)}")
    return mapping, has_vid

def annotate_candidates(mapping, has_vid):
    log("Annotating candidates file and writing output: " + args.out)
    ds_cand = ds.dataset(args.candidates, format="parquet")
    # choose columns from candidates
    cand_cols = ds_cand.schema.names
    # we'll add country_code column
    writer = None
    scanner = ds_cand.scanner(batch_size=args.batch_size)
    batches_written = 0
    total_rows = 0
    assigned = 0
    for batch in scanner.to_batches():
        n = batch.num_rows
        total_rows += n
        # build country list for this batch
        country_list = []
        for i in range(n):
            try:
                if has_vid and "video_id" in batch.schema.names:
                    vid = batch.column(batch.schema.get_field_index("video_id"))[i].as_py()
                    c = mapping.get(vid)
                    if c is None and args.use_signature_if_missing:
                        # fallback try signature
                        title = batch.column(batch.schema.get_field_index("title"))[i].as_py() if "title" in batch.schema.names else ""
                        channel = batch.column(batch.schema.get_field_index("channelTitle"))[i].as_py() if "channelTitle" in batch.schema.names else ""
                        pub = batch.column(batch.schema.get_field_index("publish_time"))[i].as_py() if "publish_time" in batch.schema.names else ""
                        s = hashlib.sha1(((title or "") + "|" + (channel or "") + "|" + (str(pub) or "")).encode("utf8")).hexdigest()
                        c = mapping.get(s)
                else:
                    # use signature key from candidates
                    title = batch.column(batch.schema.get_field_index("title"))[i].as_py() if "title" in batch.schema.names else ""
                    channel = batch.column(batch.schema.get_field_index("channelTitle"))[i].as_py() if "channelTitle" in batch.schema.names else ""
                    pub = batch.column(batch.schema.get_field_index("publish_time"))[i].as_py() if "publish_time" in batch.schema.names else ""
                    s = hashlib.sha1(((title or "") + "|" + (channel or "") + "|" + (str(pub) or "")).encode("utf8")).hexdigest()
                    c = mapping.get(s)
            except Exception:
                c = None
            if c is None:
                country_list.append("XX")
            else:
                country_list.append(str(c))
                assigned += 1
        # append column
        country_arr = pa.array(country_list)
        new_batch = batch.append_column("country_code", country_arr)
        # convert the RecordBatch to a Table (works on all pyarrow versions)
        tbl = pa.Table.from_batches([new_batch])

        if writer is None:
            writer = pq.ParquetWriter(args.out, tbl.schema)
        writer.write_table(tbl)

        batches_written += 1
        if batches_written % 10 == 0:
            log(f"wrote {batches_written} batches, total_rows={total_rows}, assigned={assigned}")
    if writer:
        writer.close()
    log(f"Done annotate: total_rows={total_rows}, assigned={assigned}")

def main():
    mapping, has_vid = build_map_from_all()
    annotate_candidates(mapping, has_vid)

if __name__ == "__main__":
    main()
