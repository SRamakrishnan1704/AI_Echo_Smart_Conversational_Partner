# 💬 Smartest Conversational Partner
### Sentiment Analysis Dashboard — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![BERT](https://img.shields.io/badge/BERT-Embeddings-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

The **Smartest Conversational Partner** is an end-to-end machine learning project that automatically classifies app reviews into three sentiment categories:

- 😊 **Positive** — Rating >= 4
- 😐 **Neutral** — Rating == 3
- 😞 **Negative** — Rating <= 2

The project includes a full NLP pipeline from raw data to a live interactive Streamlit dashboard with 20+ visualizations.

---

## 🗂️ Project Structure

```
Smartest_Conversational_Partner/
│
├── App/
│   ├── app.py                        # Streamlit dashboard (7 pages)
│   └── SCP_Pipeline.png              # Pipeline flowchart image
│
├── Data/
│   ├── chatgpt_style_reviews_dataset.xlsx   # Raw dataset
│   └── cleaned_data.csv                      # Cleaned dataset
│
├── Model/
│   ├── lr_tfidf_model.pkl            # Best model — LR + TF-IDF
│   ├── lr_bert_model.pkl             # LR + BERT
│   ├── rf_tfidf_model.pkl            # Random Forest + TF-IDF
│   ├── rf_bert_model.pkl             # Random Forest + BERT
│   ├── lstm_bert_model.keras         # LSTM + BERT
│   ├── onehot_encoder.pkl            # One-Hot encoder
│   ├── label_encoder_version.pkl     # Version encoder
│   └── label_encoder_verified.pkl    # Verified purchase encoder
│
├── pklfiles/
│   └── PKL_Outputs/
│       ├── label_encoder.pkl         # Sentiment label encoder
│       ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│       ├── scaler_tfidf.pkl          # StandardScaler for TF-IDF
│       ├── scaler_bert.pkl           # StandardScaler for BERT
│       ├── X_tfidf.npy               # TF-IDF feature matrix
│       ├── X_tfidf_scaled.npy        # Scaled TF-IDF features
│       ├── X_bert.npy                # BERT embeddings
│       ├── X_bert_scaled.npy         # Scaled BERT embeddings
│       └── y_sentiment.npy           # Target labels
│
├── Src/
│   ├── loader.py                     # Step 1 — Data loading
│   ├── Data_Cleaning.py              # Step 2 — Data cleaning
│   ├── eda.py                        # Step 3 — Exploratory data analysis
│   ├── Featureextraction_1.py        # Step 4 — Feature extraction
│   └── Model_train.py                # Step 5 — Model training
│
├── Step3_Prediction.py               # Prediction script with fallback
├── pipeline_chart.py                 # Pipeline flowchart generator
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Pipeline Architecture

```
Raw Data (CSV)
      ↓
loader.py          — Load raw dataset
      ↓
Data_Cleaning.py   — Remove nulls, clean text, normalize
      ↓
eda.py             — Exploratory analysis and visualizations
      ↓
Featureextraction_1.py
   ├── TF-IDF Vectorizer (5,000 features, bigrams)
   └── BERT Embeddings  (768-dim, CLS token)
      ↓
Model_train.py
   ├── Logistic Regression + TF-IDF  ✅ Best
   ├── Logistic Regression + BERT
   ├── Random Forest + TF-IDF
   ├── Random Forest + BERT
   └── LSTM + BERT
      ↓
Step3_Prediction.py  — Predict new reviews
      ↓
app.py               — Streamlit dashboard
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Reviews | 500 |
| Positive | 200 (40%) |
| Negative | 195 (39%) |
| Neutral | 105 (21%) |
| Average Rating | 3.01 |
| Features | review, rating, helpful_votes, review_length, platform, language, location, version, verified_purchase |

---

## 🤖 Models Trained

| Model | Features | Accuracy | Size | Speed |
|---|---|---|---|---|
| **LR + TF-IDF** ✅ | TF-IDF + categorical | 100%* | 4.0 KB | Fastest |
| LR + BERT | BERT + categorical | 100%* | 19.4 KB | Fast |
| RF + TF-IDF | TF-IDF + categorical | 100%* | 657 KB | Slow |
| RF + BERT | BERT + categorical | 100%* | 247 KB | Slow |
| LSTM + BERT | Raw BERT embeddings | 100%* | 5.38 MB | Slowest |

> ⚠️ *100% accuracy is due to synthetic data with only 112 unique words. Real-world accuracy would be 75–85%.

**Production Model:** Logistic Regression + TF-IDF — chosen for speed, size, and interpretability.

---

## 🖥️ Streamlit Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | KPI cards, rating distribution, sentiment pie chart |
| 🔮 Predict Sentiment | Live prediction with confidence scores |
| 📊 EDA Dashboard | Helpful votes, keywords, review length analysis |
| 🌍 Location Analysis | Regional sentiment breakdown |
| 📱 Platform & Version | Platform comparison, version ratings |
| 🧠 Sentiment Analysis Q&A | 10 key questions with interactive charts |
| 🔁 Project Pipeline | End-to-end pipeline flowchart |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smartest-conversational-partner.git
cd smartest-conversational-partner
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Feature Extraction
```bash
python Src/Featureextraction_1.py
```

### 4. Train Models
```bash
python Src/Model_train.py
```

### 5. Launch Dashboard
```bash
cd App
streamlit run app.py
```

### 6. Run Prediction Script
```bash
python Step3_Prediction.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
transformers
torch
keras
tensorflow
plotly
matplotlib
Pillow
pickle-mixin
```

Install all at once:
```bash
pip install streamlit pandas numpy scikit-learn transformers torch keras tensorflow plotly matplotlib Pillow
```

---

## 🔮 How Prediction Works

```python
from Step3_Prediction import predict_sentiment

result = predict_sentiment(
    review     = "This app is absolutely amazing!",
    platform   = "Web",
    language   = "English",
    location   = "India",
    version    = "2.1.4",
    verified_purchase = "Yes",
    helpful_votes = 10
)

print(result['sentiment'])    # Positive
print(result['confidence'])   # 87.5%
print(result['class_scores']) # {'Positive': 87.5, 'Neutral': 7.2, 'Negative': 5.3}
```

**Fallback logic:** When model confidence is below 60%, a keyword-based fallback checks for positive words (amazing, great, excellent) and negative words (bug, crash, terrible) to ensure robust predictions for out-of-vocabulary inputs.

---

## 🧠 Key Technologies

| Technology | Purpose |
|---|---|
| **BERT** (bert-base-uncased) | Extract 768-dim semantic embeddings from review text |
| **TF-IDF** | Extract keyword frequency features from review text |
| **Logistic Regression** | Final production classifier |
| **LSTM** | Deep learning sequence classifier using BERT features |
| **Random Forest** | Ensemble tree-based classifier |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts and visualizations |
| **HuggingFace Transformers** | BERT tokenizer and model |

---

## ⚠️ Known Limitations

1. **Synthetic Data** — Dataset is artificially generated with only 500 rows and 112 unique words
2. **100% Accuracy** — Result of synthetic data, not real-world performance
3. **Data Leakage** — Originally present (rating → sentiment), fixed by dropping rating before extraction
4. **No Date Column** — Time trend analysis uses batch index simulation instead of real dates
5. **Small Vocabulary** — Out-of-vocabulary words use keyword fallback for prediction

---

## 🚀 Future Improvements

- [ ] Add 5,000+ real app store reviews from Google Play / App Store
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Add real-time review scraping
- [ ] Fine-tune BERT on domain-specific data
- [ ] Build REST API using FastAPI
- [ ] Add Docker containerization
- [ ] Implement MLflow experiment tracking

---

## 👨‍💻 Author

**S Ramakrishnan**
Data Scientist
GUVI — Data Science Program
Date: March 2026

---

## 📄 License

This project is licensed under the MIT License.

---

> Built with ❤️ using Python, Streamlit, BERT, and scikit-learn