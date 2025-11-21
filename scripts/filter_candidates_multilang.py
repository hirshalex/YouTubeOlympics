# scripts/filter_candidates_stream_batches.py
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import re
import argparse, os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/processed/ALL_youtube_trending_data.parquet")
parser.add_argument("--output", default="data/processed/candidates.parquet")
parser.add_argument("--batch_size", type=int, default=2000, help="rows per record-batch")
parser.add_argument("--min_samples", type=int, default=0)
args = parser.parse_args()

LOG_DIR = "logs"
PROGRESS_FILE = os.path.join(LOG_DIR, "progress_filter_candidates.txt")
os.makedirs(LOG_DIR, exist_ok=True)
open(PROGRESS_FILE, "a").close()

def log(s):
    ts = datetime.utcnow().isoformat()+"Z"
    line = f"{ts} {s}"
    print(line, flush=True)
    with open(PROGRESS_FILE, "a") as fh:
        fh.write(line + "\n")

KEYWORDS = [
  "olymp","medal","athlet","gymnast","swim","swimmer","track","field","marathon",
  "football","soccer","tennis","basket","basketball","baseball","cricket","hockey",
  "volley","volleyball","karate","judo","taekwondo","rowing","canoe","kayak",
  "sprint","relay","fencing","weightlift","boxing","mma","surf","skate","skating",
  "ski","snowboard",
  "olimpi","olímp","medalla","atleta","natación","maratón","fútbol","baloncesto","voleibol",
  "medalha","atleta","natação","maratona","futebol","voleibol","remo","ginástica",
  "médaille","athlète","natation","gymnastique","marathon",
  "medaille","leichtathletik","schwimm","turnen","fußball",
  "olimp","medaglia","atleta","nuot","ginnastica","maratona","calcio",
  "олимп","медал","спорт","атлет","гимнаст","бег","плаван","футбол","теннис","бокс",
  "올림","메달","선수","체조","수영","달리기","축구","농구","테니스",
  "オリンピ","メダル","選手","体操","水泳","マラソン","サッカー","野球","テニス",
  "khiladi","खिलाड़ी","दौड़","मैरा","क्रिकेट"
]

pattern = "|".join([re.escape(k) for k in KEYWORDS])  # regex string; pandas will handle flags


def main():
    log(f"START streaming filter from {args.input} batch_size={args.batch_size}")
    if not os.path.exists(args.input):
        log(f"ERROR input not found: {args.input}")
        raise SystemExit(1)

    dataset = ds.dataset(args.input, format="parquet")
    # Determine available columns to reduce memory
    available_cols = [c for c in ["title","tags","description","channelTitle","category_id","country_code"] if c in dataset.schema.names]
    log(f"Using columns: {available_cols}")
    scanner = dataset.scanner(columns=available_cols or None, batch_size=args.batch_size)

    writer = None
    total_matches = 0
    batch_i = 0
    try:
        for batch in scanner.to_batches():
            batch_i += 1
            try:
                df = batch.to_pandas()
            except Exception as e:
                log(f"ERROR converting batch to pandas (batch {batch_i}): {e}")
                continue
            # build combined text column (vectorized)
            if available_cols:
                combined = df[available_cols[0]].fillna("").astype(str)
                for c in available_cols[1:]:
                    combined = combined.str.cat(df[c].fillna("").astype(str), sep=" ")
            else:
                combined = pd.Series([""] * len(df))
            mask = combined.str.contains(pattern, case=False, regex=True, na=False)
            matched = df[mask]
            if len(matched):
                total_matches += len(matched)
                table = pa.Table.from_pandas(matched)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, table.schema)
                    writer.write_table(table)
                else:
                    writer.write_table(table)
            if batch_i % 10 == 0:
                log(f"batches={batch_i} total_matches={total_matches}")
    except Exception as e:
        log(f"ERROR in scanning loop: {e}")
        raise
    finally:
        if writer:
            writer.close()
    log(f"DONE batches={batch_i} total_matches={total_matches}")

    # optional subsample afterwards (if requested)
    if args.min_samples and total_matches > args.min_samples:
        log(f"SUBSAMPLE to {args.min_samples}")
        import pandas as pd
        dfm = pd.read_parquet(args.output)
        dfm.sample(n=args.min_samples, random_state=0).to_parquet(args.output, index=False)
        log("SUBSAMPLED done")

if __name__ == "__main__":
    main()
