import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing           import LabelEncoder, StandardScaler
from transformers                    import BertTokenizer, BertModel
import torch

# ─────────────────────────────────────────────────────────────────
# CONFIG — Paths
# ─────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Data\cleaned_data.csv"
OUTPUT_DIR = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\pklfiles\PKL_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BERT_BATCH = 32
MAX_LEN    = 128

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load Cleaned Data
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1 — Load Cleaned Data")
print("=" * 65)

data = pd.read_csv(DATA_PATH)
print(f"  Rows Loaded         : {len(data)}")
print(f"  Columns Available   : {list(data.columns)}")

# ── Add sentiment column from rating ─────────────────────────────
def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

data['sentiment'] = data['rating'].apply(map_sentiment)
print(f"  Sentiment column added ✅")
print(f"  Sentiment Counts    :\n{data['sentiment'].value_counts().to_string()}")

# ── Check data quality — sample reviews per sentiment ────────────
print(f"\n  Sample Reviews per Sentiment:")
for sentiment in ['Positive', 'Neutral', 'Negative']:
    samples = data[data['sentiment'] == sentiment]['review'].head(3).tolist()
    print(f"\n  {sentiment}:")
    for s in samples:
        print(f"    → {s}")

# ── Safety check — drop leftover unwanted columns if present ─────
cols_to_drop = ['date', 'title', 'username', 'tokens']
dropped      = [c for c in cols_to_drop if c in data.columns]
if dropped:
    data = data.drop(columns=dropped)
    print(f"  Safety drop         : {dropped}")
else:
    print(f"  No extra columns to drop ✅")

print(f"  Final Columns       : {list(data.columns)}")
print(f"\n  Null Values         :\n{data.isnull().sum().to_string()}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Label Encode Target (sentiment)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2 — Label Encoding  (Target: sentiment)")
print("=" * 65)

le = LabelEncoder()
y  = le.fit_transform(data['sentiment'])

print(f"  Classes             : {list(le.classes_)}")
print(f"\n  Encoding Map        :")
for label, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"    {label:<15} → {code}")

print(f"\n  Class Distribution  :")
for label, code in zip(le.classes_, le.transform(le.classes_)):
    count = (y == code).sum()
    pct   = count / len(y) * 100
    print(f"    {label:<15} : {count} samples  ({pct:.1f}%)")

# Save
with open(f"{OUTPUT_DIR}/label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)
np.save(f"{OUTPUT_DIR}/y_sentiment.npy", y)
print("\n  Label encoder saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — TF-IDF Vectorization
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3 — TF-IDF Vectorization")
print("=" * 65)

tfidf = TfidfVectorizer(
    max_features  = 5000,
    ngram_range   = (1, 2),     # unigrams + bigrams
    sublinear_tf  = True,       # log normalization
    strip_accents = 'unicode',
    analyzer      = 'word',
    min_df        = 1           # ignore very rare terms
)

X_tfidf = tfidf.fit_transform(data['review']).toarray()
print(f"  TF-IDF Matrix Shape : {X_tfidf.shape}")
print(f"  Vocabulary Size     : {len(tfidf.vocabulary_)}")
print(f"  Top 10 Features     : {tfidf.get_feature_names_out()[:10].tolist()}")

# Scale TF-IDF
scaler_tfidf   = StandardScaler()
X_tfidf_scaled = scaler_tfidf.fit_transform(X_tfidf)
print(f"  Scaled Shape        : {X_tfidf_scaled.shape}")

# Save
with open(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(tfidf, f)
with open(f"{OUTPUT_DIR}/scaler_tfidf.pkl", 'wb') as f:
    pickle.dump(scaler_tfidf, f)
np.save(f"{OUTPUT_DIR}/X_tfidf.npy",        X_tfidf)
np.save(f"{OUTPUT_DIR}/X_tfidf_scaled.npy", X_tfidf_scaled)
print("  TF-IDF vectorizer & scaler saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — BERT Embeddings
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4 — BERT Embeddings")
print("=" * 65)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model     = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
print(f"  BERT Loaded ✅  |  Device : {device}")

def get_bert_embeddings_batch(texts, tokenizer, model, device,
                               batch_size=BERT_BATCH, max_length=MAX_LEN):
    """Batch BERT CLS-token embedding extraction."""
    all_embeddings = []
    texts = list(texts)
    total = len(texts)
    for i in range(0, total, batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors = 'pt',
            max_length     = max_length,
            truncation     = True,
            padding        = 'max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs    = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embeddings.append(embeddings.cpu().numpy())
        done = min(i + batch_size, total)
        print(f"  Processed : {done}/{total} reviews...", end='\r')
    print()
    return np.vstack(all_embeddings)

print("\n  Generating BERT embeddings (this may take a few minutes)...")
X_bert = get_bert_embeddings_batch(
    data['review'], bert_tokenizer, bert_model, device
)
print(f"  BERT Matrix Shape   : {X_bert.shape}")

# Scale BERT
scaler_bert   = StandardScaler()
X_bert_scaled = scaler_bert.fit_transform(X_bert)

# Save
np.save(f"{OUTPUT_DIR}/X_bert.npy",        X_bert)
np.save(f"{OUTPUT_DIR}/X_bert_scaled.npy", X_bert_scaled)
with open(f"{OUTPUT_DIR}/X_bert.pkl", 'wb') as f:
    pickle.dump(X_bert, f)
with open(f"{OUTPUT_DIR}/scaler_bert.pkl", 'wb') as f:
    pickle.dump(scaler_bert, f)
print("  BERT embeddings & scaler saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Final Summary of All Saved PKL Files
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5 — Saved PKL / NPY Files Summary")
print("=" * 65)

saved_files = [
    ("label_encoder.pkl",    "Target label encoder (sentiment classes)"),
    ("y_sentiment.npy",      "Encoded target labels (y)"),
    ("tfidf_vectorizer.pkl", "TF-IDF vectorizer (fit on review text)"),
    ("scaler_tfidf.pkl",     "StandardScaler for TF-IDF features"),
    ("X_tfidf.npy",          "TF-IDF feature matrix (raw)"),
    ("X_tfidf_scaled.npy",   "TF-IDF feature matrix (scaled)"),
    ("X_bert.pkl",           "BERT embeddings (pkl format)"),
    ("X_bert.npy",           "BERT embeddings (numpy format)"),
    ("scaler_bert.pkl",      "StandardScaler for BERT features"),
    ("X_bert_scaled.npy",    "BERT embeddings (scaled)"),
]

print(f"\n  {'File':<30} {'Size':>10}   Description")
print(f"  {'─' * 70}")
for fname, desc in saved_files:
    path = f"{OUTPUT_DIR}/{fname}"
    if os.path.exists(path):
        size     = os.path.getsize(path)
        size_str = (f"{size/1024/1024:.2f} MB" if size > 1024*1024
                    else f"{size/1024:.1f} KB")
        print(f"  ✅ {fname:<28} {size_str:>10}   {desc}")
    else:
        print(f"  ❌ {fname:<28} {'N/A':>10}   {desc}")

print(f"\n  Total Samples       : {len(data)}")
print(f"  TF-IDF Features     : {X_tfidf.shape[1]}")
print(f"  BERT Features       : {X_bert.shape[1]}")
print(f"  Target Classes      : {list(le.classes_)}")
print(f"\n  ✅ Feature Extraction Complete!")
print(f"  📁 All PKL files saved to → {OUTPUT_DIR}")
print(f"\n  ➡️  Now run  Step2_Model_Training.py  to train models!")

# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Feature Extraction Completed ✅ ---")
    print(f"  Total Samples  : {X_bert.shape[0]}")
    print(f"  TF-IDF Shape   : {X_tfidf_scaled.shape}")
    print(f"  BERT Shape     : {X_bert_scaled.shape}")
    print(f"  Target Shape   : {y.shape}")