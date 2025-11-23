# scripts/extract_olympic_candidates_ext.py
import pandas as pd, re
from pathlib import Path

INFILE = "data/processed/ALL_youtube_trending_data.parquet"  # adjust if different
OUTFILE = "data/processed/olympic_candidates.parquet"
PER_COUNTRY_CAP = 200   # how many per country to keep; increase if you want more
GLOBAL_CAP = 2000       # overall cap if you want to limit size

# multilingual stems and extra sport/medal keywords
stems = [
 "olymp", "olimpi", "olímp", "olimpiá", "olympiad", "olympiade", "olympics",
 "올림픽", "オリンピック", "олимп", "奥运", "奥林匹克", "olympische", "olimpiadi",
 # medalish/indicator words that often co-occur with Olympic articles
 "medal", "medalha", "medalla", "medaglia", "podium", "podio", "gold", "silver", "bronze"
]
pattern = re.compile("|".join(re.escape(s) for s in stems), flags=re.I)

def text_for_row(r):
    # join title, tags, description, channel_title (if present)
    return " ".join([str(r.get(c,"") or "") for c in ("title","tags","description","channel_title")])

def main():
    print("Loading master file:", INFILE)
    df = pd.read_parquet(INFILE)
    print("Master rows:", len(df))
    # filter by pattern
    mask = df.apply(lambda r: bool(pattern.search(text_for_row(r))), axis=1)
    cands = df[mask].copy()
    print("Candidates after stem/keyword filter:", len(cands))

    out = []
    if 'country_code' in cands.columns:
        for country, g in cands.groupby('country_code'):
            keep = g.head(PER_COUNTRY_CAP)
            out.append(keep)
    else:
        out.append(cands.head(GLOBAL_CAP))

    sampled = pd.concat(out, ignore_index=True)
    if GLOBAL_CAP and len(sampled) > GLOBAL_CAP:
        sampled = sampled.sample(n=GLOBAL_CAP, random_state=42).reset_index(drop=True)

    print("Final sampled candidate rows:", len(sampled))
    Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(OUTFILE, index=False)
    print("Saved:", OUTFILE)

if __name__ == "__main__":
    main()
