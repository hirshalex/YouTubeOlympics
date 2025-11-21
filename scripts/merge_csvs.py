# scripts/merge_csvs.py
import pandas as pd
import glob, os

csvs = sorted(glob.glob("data/raw/*_youtube_trending_data.csv"))
dfs = []
for f in csvs:
    country = os.path.basename(f).split("_")[0]  # BR_youtube_trending_data.csv -> BR
    df = pd.read_csv(f, low_memory=False)
    df['country_code'] = country
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_parquet("data/processed/ALL_youtube_trending_data.parquet", index=False)  # parquet is faster
print("Saved", len(df_all), "rows to data/processed/ALL_youtube_trending_data.parquet")