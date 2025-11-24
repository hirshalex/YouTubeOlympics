# scripts/train_student_tfidf_improved.py
import os
import pandas as pd
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

FILES = [
    "data/processed/groq_labeled_sampled_pos.csv",
    "data/processed/groq_labeled_sampled_neg.csv",
    "data/processed/groq_labeled_olympic_candidates.csv",
]

# parameters you can tune:
UPSAMPLE_TARGET = 400        # target number of olympic examples after upsampling
MAX_FEATURES = 80000
NGRAM_RANGE = (1,3)          # word n-grams
MIN_DF = 1                   # allow singletons (helps tiny class)
CHAR_NGRAMS = False          # set True to add char ngrams later if needed
RANDOM_STATE = 42

print("Loading files...")
dfs = []
for f in FILES:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
    else:
        print(f"[WARN] missing {f}")
if not dfs:
    raise SystemExit("No labeled CSVs found.")

df = pd.concat(dfs, ignore_index=True)

# robust merge_key dedupe
df["video_id"] = df.get("video_id")
df["merge_key"] = (
    df.get("video_id").fillna("").astype(str).str.strip() + "||" +
    df.get("title").fillna("").astype(str).str.strip() + "||" +
    df.get("tags").fillna("").astype(str).str.strip()
)
df = df.drop_duplicates(subset=["merge_key"]).reset_index(drop=True)

df['label'] = df['label'].astype(str).str.lower().str.strip()
df['text'] = df['title'].fillna("") + " " + df['tags'].fillna("") + " " + df['description'].fillna("")

print("Counts before upsampling:", df['label'].value_counts())

# separate classes
ol = df[df['label']=="olympic"]
rest = df[df['label']!="olympic"]

if len(ol) == 0:
    raise SystemExit("No olympic examples found; upsampling impossible.")

# upsample olympic by sampling with replacement to reach UPSAMPLE_TARGET
if len(ol) < UPSAMPLE_TARGET:
    reps = ol.sample(n=max(0, UPSAMPLE_TARGET - len(ol)), replace=True, random_state=RANDOM_STATE)
    ol_up = pd.concat([ol, reps], ignore_index=True)
else:
    ol_up = ol.sample(n=UPSAMPLE_TARGET, random_state=RANDOM_STATE)

df_trainable = pd.concat([rest, ol_up], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

print("Counts after upsampling:", df_trainable['label'].value_counts())

# vectorizer
print("Fitting TF-IDF (word ngrams, min_df=1)...")
vec = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    analyzer='word',
    min_df=MIN_DF,
    max_df=0.98
)
X = vec.fit_transform(df_trainable['text'].tolist())
y = df_trainable['label'].tolist()

# train/val split (stratify now okay)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Train label counts:", Counter(y_train))
print("Val   label counts:", Counter(y_val))

# classifier: saga is good for large sparse problems; increase max_iter
clf = LogisticRegression(
    solver="saga",
    max_iter=10000,
    class_weight="balanced",
    C=1.0,
    multi_class="multinomial",
    random_state=RANDOM_STATE,
)

print("Training classifier...")
clf.fit(X_train, y_train)

print("Evaluating...")
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred, labels=clf.classes_))

# show top features for 'olympic' class
try:
    class_idx = list(clf.classes_).index("olympic")
    coef = clf.coef_[class_idx]  # if multinomial, shape = (n_classes, n_features)
    topn = 40
    top_idxs = np.argsort(coef)[-topn:][::-1]
    feature_names = np.array(vec.get_feature_names_out())
    print("\nTop features for 'olympic' class (top {}):".format(topn))
    for i in top_idxs[:topn]:
        print(feature_names[i], f"{coef[i]:.4f}")
except Exception as e:
    print("Could not extract top features:", e)

# save
os.makedirs("models", exist_ok=True)
joblib.dump({"vec": vec, "clf": clf}, "models/tfidf_student.joblib")
print("Saved models/tfidf_student.joblib")
