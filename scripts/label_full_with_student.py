#!/usr/bin/env python3
"""
Label all YouTube videos with the trained student classifier.

Input:
  data/processed/all_videos.parquet     ← merged master dataset
Model:
  models/student_clf.joblib             ← logistic regression + embed model name
  models/embed_model/                   ← SentenceTransformer directory
Output:
  data/processed/all_videos_with_labels.parquet
"""
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

INPUT = "data/processed/all_videos.parquet"
OUTPUT = "data/processed/all_videos_with_labels.parquet"
MODEL_PATH = "models/student_clf.joblib"

print("Loading classifier and embedding model...")
bundle = joblib.load(MODEL_PATH)
clf = bundle["clf"]

embed_model_name = bundle.get("embed_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_model_dir = bundle.get("embed_model_dir", None)

# load embedding model (prefer local directory if saved)
if embed_model_dir:
    try:
        embed_model = SentenceTransformer(embed_model_dir)
        print(f"Loaded embedding model from {embed_model_dir}")
    except Exception as e:
        print(f"[WARN] failed to load from {embed_model_dir}: {e}")
        embed_model = SentenceTransformer(embed_model_name)
else:
    embed_model = SentenceTransformer(embed_model_name)

print("Loading dataset:", INPUT)
df = pd.read_parquet(INPUT)
print("Rows:", len(df))

# Build text field
df["text"] = (
    df["title"].fillna("") + " "
    + df["tags"].fillna("") + " "
    + df["description"].fillna("")
)

# Encode in batches and predict
batch_size = 5000
labels = []

print("Encoding and predicting in batches...")
for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i : i + batch_size].tolist()
    X = embed_model.encode(batch_texts, show_progress_bar=False)
    preds = clf.predict(X)
    labels.extend(preds)
    if (i + batch_size) % 50000 < batch_size:
        print(f"Processed {i + len(batch_texts)} / {len(df)} rows")

df["predicted_label"] = labels
df.to_parquet(OUTPUT, index=False)
print("✅ Saved labeled dataset ->", OUTPUT)
print(df["predicted_label"].value_counts())
