# scripts/train_student.py
import pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

pos = pd.read_csv("data/processed/groq_labeled_sampled_pos.csv")
neg = pd.read_csv("data/processed/groq_labeled_sampled_neg.csv")
olympic = pd.read_csv("data/processed/groq_labeled_olympic_candidates.csv")
df = pd.concat([pos, neg, olympic], ignore_index=True)
df['text'] = df['title'].fillna("") + " " + df['tags'].fillna("") + " " + df['description'].fillna("")
X_text = df['text'].tolist()
y = df['label'].tolist()

print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
X = embed_model.encode(X_text, show_progress_bar=True, batch_size=64)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
joblib.dump({"clf": clf, "embed_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}, "models/student_clf.joblib")
# Save embeddings model separately if you want to reuse instantly
print("Saved models/student_clf.joblib")
