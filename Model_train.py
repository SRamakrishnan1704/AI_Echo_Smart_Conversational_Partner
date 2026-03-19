import numpy as np
import pickle
import os
import keras
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     accuracy_score)
import pandas as pd

from keras.models    import Sequential
from keras.layers    import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils     import to_categorical

# ─────────────────────────────────────────────────────────────────
# CONFIG — Paths
# ─────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Data\cleaned_data.csv"
PKL_DIR    = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\pklfiles\PKL_Outputs"
OUTPUT_DIR = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load All PKL Files from Step 1
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1 — Load PKL Files")
print("=" * 65)

# Load encoders & scalers
with open(f"{PKL_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)
with open(f"{PKL_DIR}/tfidf_vectorizer.pkl", 'rb') as f:
    tfidf = pickle.load(f)
with open(f"{PKL_DIR}/scaler_tfidf.pkl", 'rb') as f:
    scaler_tfidf = pickle.load(f)
with open(f"{PKL_DIR}/scaler_bert.pkl", 'rb') as f:
    scaler_bert = pickle.load(f)

# Load feature matrices
X_tfidf_scaled = np.load(f"{PKL_DIR}/X_tfidf_scaled.npy")
X_tfidf        = np.load(f"{PKL_DIR}/X_tfidf.npy")
X_bert         = np.load(f"{PKL_DIR}/X_bert.npy")
X_bert_scaled  = np.load(f"{PKL_DIR}/X_bert_scaled.npy")
y              = np.load(f"{PKL_DIR}/y_sentiment.npy")

print(f"  Label Classes       : {list(le.classes_)}")
print(f"  TF-IDF Shape        : {X_tfidf_scaled.shape}")
print(f"  BERT Shape          : {X_bert_scaled.shape}")
print(f"  Target Shape        : {y.shape}")
print(f"  All PKL files loaded ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Encode Remaining Categorical Columns
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2 — Encode Remaining Categorical Columns")
print("=" * 65)

data = pd.read_csv(DATA_PATH)

# ── Drop rating — prevents data leakage ──────────────────────────
data = data.drop(columns=['rating'])
print(f"  Leakage column dropped  : ['rating']")
print(f"  Remaining columns       : {list(data.columns)}")

# ── Drop rating to prevent data leakage ──────────────────────────
cols_to_drop = ['rating', 'sentiment']  # sentiment is already encoded in y
dropped = [c for c in cols_to_drop if c in data.columns]
data = data.drop(columns=dropped)
print(f"  Leakage columns dropped : {dropped}")
print(f"  Remaining columns       : {list(data.columns)}")

# ── 2.1 One-Hot Encoding — platform, language, location ──────────
onehot_cols     = ['platform', 'language', 'location']
onehot_encoder  = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_onehot        = onehot_encoder.fit_transform(data[onehot_cols])
print(f"  One-Hot Shape       : {X_onehot.shape}")
print(f"  Columns encoded     : {onehot_cols}")

with open(f"{OUTPUT_DIR}/onehot_encoder.pkl", 'wb') as f:
    pickle.dump(onehot_encoder, f)
print("  One-Hot encoder saved ✅")

# ── 2.2 Label Encoding — version ─────────────────────────────────
le_version              = LabelEncoder()
X_version               = le_version.fit_transform(
                            data['version'].astype(str)).reshape(-1, 1)
print(f"\n  Version unique vals : {data['version'].nunique()}")

with open(f"{OUTPUT_DIR}/label_encoder_version.pkl", 'wb') as f:
    pickle.dump(le_version, f)
print("  Version encoder saved ✅")

# ── 2.3 Label Encoding — verified_purchase ───────────────────────
le_verified             = LabelEncoder()
X_verified              = le_verified.fit_transform(
                            data['verified_purchase'].astype(str)).reshape(-1, 1)
print(f"  verified_purchase   : {dict(zip(le_verified.classes_, le_verified.transform(le_verified.classes_)))}")

with open(f"{OUTPUT_DIR}/label_encoder_verified.pkl", 'wb') as f:
    pickle.dump(le_verified, f)
print("  Verified encoder saved ✅")    

# ── 2.4 Numeric columns — helpful_votes, review_length ───────────
X_numeric = data[['helpful_votes', 'review_length']].values
print(f"\n  Numeric Shape       : {X_numeric.shape}")

# ── 2.5 Combine categorical + numeric ────────────────────────────
X_extra = np.hstack([X_onehot, X_version, X_verified, X_numeric])
print(f"  Combined Extra Shape: {X_extra.shape}")

# ── 2.6 Combine with TF-IDF and BERT ─────────────────────────────
X_tfidf_full = np.hstack([X_tfidf_scaled, X_extra])
X_bert_full  = np.hstack([X_bert_scaled,  X_extra])
print(f"\n  TF-IDF Full Shape   : {X_tfidf_full.shape}  (TF-IDF + categorical + numeric)")
print(f"  BERT Full Shape     : {X_bert_full.shape}   (BERT + categorical + numeric)")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Train / Test Split
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3 — Train / Test Split  (80 / 20)")
print("=" * 65)

# TF-IDF split
(X_tfidf_train, X_tfidf_test,
 y_train,       y_test) = train_test_split(
    X_tfidf_full, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y
)

# BERT split — same indices
indices              = np.arange(len(y))
train_idx, test_idx  = train_test_split(
    indices,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y
)
X_bert_train     = X_bert_full[train_idx]
X_bert_test      = X_bert_full[test_idx]
X_bert_raw_train = X_bert[train_idx]
X_bert_raw_test  = X_bert[test_idx]
y_train_bert     = y[train_idx]
y_test_bert      = y[test_idx]

print(f"  Train Samples       : {len(y_train)}")
print(f"  Test  Samples       : {len(y_test)}")
print(f"  Train Distribution  : {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Test  Distribution  : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────
# HELPER — Evaluate Model
# ─────────────────────────────────────────────────────────────────
results = {}

def evaluate(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  ┌─ Results : {model_name}")
    print(f"  │  Accuracy : {acc * 100:.2f}%")
    print(f"  └─ Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names = le.classes_,
        digits       = 4
    ))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {cm}")
    results[model_name] = round(acc * 100, 2)
    return acc

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Logistic Regression
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4 — Logistic Regression")
print("=" * 65)

# ── 4a. LR + TF-IDF ──────────────────────────────────────────────
print("\n  Training LR + TF-IDF...")
lr_tfidf = LogisticRegression(
    max_iter     = 1000,
    class_weight = 'balanced',
    solver       = 'lbfgs',
    random_state = RANDOM_STATE
)
lr_tfidf.fit(X_tfidf_train, y_train)
evaluate("LR + TF-IDF", y_test, lr_tfidf.predict(X_tfidf_test))

with open(f"{OUTPUT_DIR}/lr_tfidf_model.pkl", 'wb') as f:
    pickle.dump(lr_tfidf, f)
print("  LR + TF-IDF model saved ✅")

# ── 4b. LR + BERT ────────────────────────────────────────────────
print("\n  Training LR + BERT...")
lr_bert = LogisticRegression(
    max_iter     = 1000,
    class_weight = 'balanced',
    solver       = 'lbfgs',
    random_state = RANDOM_STATE
)
lr_bert.fit(X_bert_train, y_train_bert)
evaluate("LR + BERT", y_test_bert, lr_bert.predict(X_bert_test))

with open(f"{OUTPUT_DIR}/lr_bert_model.pkl", 'wb') as f:
    pickle.dump(lr_bert, f)
print("  LR + BERT model saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Random Forest
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5 — Random Forest")
print("=" * 65)

# ── 5a. RF + TF-IDF ──────────────────────────────────────────────
print("\n  Training RF + TF-IDF  (may take a few minutes)...")
rf_tfidf = RandomForestClassifier(
    n_estimators = 200,
    class_weight = 'balanced',
    random_state = RANDOM_STATE,
    n_jobs       = -1
)
rf_tfidf.fit(X_tfidf_train, y_train)
evaluate("RF + TF-IDF", y_test, rf_tfidf.predict(X_tfidf_test))

with open(f"{OUTPUT_DIR}/rf_tfidf_model.pkl", 'wb') as f:
    pickle.dump(rf_tfidf, f)
print("  RF + TF-IDF model saved ✅")

# ── 5b. RF + BERT ────────────────────────────────────────────────
print("\n  Training RF + BERT  (may take a few minutes)...")
rf_bert = RandomForestClassifier(
    n_estimators = 200,
    class_weight = 'balanced',
    random_state = RANDOM_STATE,
    n_jobs       = -1
)
rf_bert.fit(X_bert_raw_train, y_train_bert)
evaluate("RF + BERT", y_test_bert, rf_bert.predict(X_bert_raw_test))

with open(f"{OUTPUT_DIR}/rf_bert_model.pkl", 'wb') as f:
    pickle.dump(rf_bert, f)
print("  RF + BERT model saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — LSTM
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 6 — LSTM  (BERT embeddings as input)")
print("=" * 65)

# Reshape for LSTM → (samples, timesteps=1, features=768)
X_lstm_train = X_bert_raw_train.reshape(
    X_bert_raw_train.shape[0], 1, X_bert_raw_train.shape[1])
X_lstm_test  = X_bert_raw_test.reshape(
    X_bert_raw_test.shape[0],  1, X_bert_raw_test.shape[1])

num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train_bert, num_classes)
y_test_cat  = to_categorical(y_test_bert,  num_classes)

lstm_model = Sequential([
    LSTM(128, input_shape=(1, 768), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name="LSTM_Sentiment_Classifier")

lstm_model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
lstm_model.summary()

early_stop = EarlyStopping(
    monitor              = 'val_loss',
    patience             = 3,
    restore_best_weights = True,
    verbose              = 1
)

print("\n  Training LSTM...")
history = lstm_model.fit(
    X_lstm_train, y_train_cat,
    validation_split = 0.1,
    epochs           = 20,
    batch_size       = 32,
    callbacks        = [early_stop],
    verbose          = 1
)

y_pred_lstm = np.argmax(lstm_model.predict(X_lstm_test), axis=1)
evaluate("LSTM + BERT", y_test_bert, y_pred_lstm)

lstm_model.save(f"{OUTPUT_DIR}/lstm_bert_model.keras")
print("  LSTM model saved ✅")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — Final Comparison Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 7 — Model Comparison Summary")
print("=" * 65)

print(f"\n  {'Rank':<6} {'Model':<25} {'Accuracy':>10}")
print(f"  {'─' * 44}")
for rank, (model_name, acc) in enumerate(
        sorted(results.items(), key=lambda x: -x[1]), start=1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
    print(f"  {medal}  {rank:<4} {model_name:<25} {acc:>9.2f}%")

best = max(results, key=results.get)
print(f"\n  🏆 Best Model        : {best}")
print(f"  🎯 Best Accuracy     : {results[best]:.2f}%")

# ─────────────────────────────────────────────────────────────────
# STEP 8 — Saved Model Files Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 8 — Saved Model Files")
print("=" * 65)

model_files = [
    ("onehot_encoder.pkl",         "One-Hot encoder (platform, language, location)"),
    ("label_encoder_version.pkl",  "Label encoder for version"),
    ("label_encoder_verified.pkl", "Label encoder for verified_purchase"),
    ("lr_tfidf_model.pkl",         "Logistic Regression + TF-IDF"),
    ("lr_bert_model.pkl",          "Logistic Regression + BERT"),
    ("rf_tfidf_model.pkl",         "Random Forest + TF-IDF"),
    ("rf_bert_model.pkl",          "Random Forest + BERT"),
    ("lstm_bert_model.keras",      "LSTM + BERT Embeddings"),
]

print(f"\n  {'File':<35} {'Size':>10}   Description")
print(f"  {'─' * 75}")
for fname, desc in model_files:
    path = f"{OUTPUT_DIR}/{fname}"
    if os.path.exists(path):
        size     = os.path.getsize(path)
        size_str = (f"{size/1024/1024:.2f} MB" if size > 1024*1024
                    else f"{size/1024:.1f} KB")
        print(f"  ✅ {fname:<33} {size_str:>10}   {desc}")
    else:
        print(f"  ❌ {fname:<33} {'N/A':>10}   {desc}")

print(f"\n  ✅ Model Training Complete!")
print(f"  📁 All models saved to → {OUTPUT_DIR}")
print(f"\n  ➡️  Now run  Step3_Prediction.py  to predict on new reviews!")