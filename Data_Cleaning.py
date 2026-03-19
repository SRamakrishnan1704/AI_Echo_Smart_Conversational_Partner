import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
 
from loader import file_path, load_data
 
data = load_data(file_path)
print("Initial Data Shape:", data.shape)
 
def clean_data(df):
    df = df.copy()
 
    # ── Describe the data ────────────────────────────────────────
    print("Data Description:")
    print(df.describe())
 
    # ── Missing values ───────────────────────────────────────────
    print("Missing Values:")
    print(df.isnull().sum())
 
    # ── Drop rows where review is missing ────────────────────────
    df = df.dropna(subset=['review'])
    print("Rows with missing 'review' dropped.")
 
    # ── Drop columns not needed for modelling ────────────────────
    cols_to_drop = ['date', 'title', 'username']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"Dropped columns: {cols_to_drop}")
    
    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    print(f"Whitespace stripped from string columns: {list(str_cols)}")
 
       # ── Convert review to lowercase ──────────────────────────────
    df['review'] = df['review'].str.lower()
    print("Converted 'review' column to lowercase.")
 
    # ── Remove special characters ────────────────────────────────
    df['review'] = df['review'].str.replace(r'[^\w\s]', '', regex=True)
    print("Special characters removed from 'review' column.")
 
    # ── Remove stop words ────────────────────────────────────────
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(
        lambda x: ' '.join(
            [word for word in x.split() if word not in stop_words]
        ) if pd.notnull(x) else x
    )
    print("Stop words removed from 'review' column.")
 
    # ── Tokenization (temporary — NOT saved as a column) ─────────
    tokens = df['review'].apply(
        lambda x: word_tokenize(x) if pd.notnull(x) else []
    )
    print("Tokenization completed.")
 
    # ── Lemmatization (temporary — NOT saved as a column) ────────
    lemmatizer = WordNetLemmatizer()
    tokens = tokens.apply(
        lambda t: [lemmatizer.lemmatize(word) for word in t]
    )
    print("Lemmatization completed.")

    # ── Join tokens back into review string ──────────────────────
    df['review'] = tokens.apply(lambda t: ' '.join(t))
    print("Tokens joined back into 'review' column.")
    # Note: tokens variable is discarded here — never added to df
    
    save_path = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Data\cleaned_data.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned data saved to → {save_path} ✅")
 
    return df
 
 
data = clean_data(data)
 
if __name__ == "__main__":
    print("\nCleaned Data Shape   :", data.shape)
    print("\nCleaned Columns      :", list(data.columns))
    