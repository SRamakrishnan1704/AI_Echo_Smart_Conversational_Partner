import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from loader import file_path, load_data
from Data_Cleaning import clean_data

# ────────────────────────────────────────────────────────────────
# STEP 1 — Map Rating to Sentiment
# ────────────────────────────────────────────────────────────────
def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# ────────────────────────────────────────────────────────────────
# STEP 2 — Sentiment Distribution (Bar Chart)
# ────────────────────────────────────────────────────────────────
def plot_sentiment_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sentiment', data=data, hue='sentiment',
              palette='Set2', order=['Positive', 'Neutral', 'Negative'],
              legend=False)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    print("Sentiment Distribution chart displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 3 — Rating Distribution (Histogram)
# ────────────────────────────────────────────────────────────────
def plot_rating_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.histplot(data['rating'], bins=5, kde=False, color='steelblue')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.show()
    print("Rating Distribution chart displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 4 — Word Cloud (All Reviews)
# ────────────────────────────────────────────────────────────────
def plot_wordcloud_all(data):
    all_words = ' '.join(data['review'].dropna())

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=200
    ).generate(all_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud — All Reviews', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("Word Cloud (All Reviews) displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 5 — Word Cloud by Sentiment
# ────────────────────────────────────────────────────────────────
def plot_wordcloud_by_sentiment(data):
    colormaps = {
        'Positive': 'Greens',
        'Neutral' : 'Blues',
        'Negative': 'Reds'
    }

    for sentiment in ['Positive', 'Neutral', 'Negative']:
        words = ' '.join(
            data[data['sentiment'] == sentiment]['review'].dropna()
        )

        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=colormaps[sentiment],
            max_words=150
        ).generate(words)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud — {sentiment} Reviews', fontsize=16)
        plt.tight_layout()
        plt.show()

    print("Word Cloud (By Sentiment) displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 6 — Top 20 Most Frequent Words (Histogram)
# ────────────────────────────────────────────────────────────────
def plot_top_words(data, top_n=20):
    all_tokens = [word for review in data['review'].dropna()
                  for word in review.split()]

    word_freq  = Counter(all_tokens).most_common(top_n)
    words_list, counts = zip(*word_freq)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words_list), hue=list(words_list),
            palette='Blues_r', legend=False)
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.show()
    print(f"Top {top_n} Most Frequent Words chart displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 7 — Top 10 Words per Sentiment (Side by Side)
# ────────────────────────────────────────────────────────────────
def plot_top_words_by_sentiment(data, top_n=10):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sentiments = ['Positive', 'Neutral', 'Negative']
    colors     = ['green', 'blue', 'red']

    for ax, sentiment, color in zip(axes, sentiments, colors):
        tokens = [word
                  for review in data[data['sentiment'] == sentiment]['review'].dropna()
                  for word in review.split()]

        freq              = Counter(tokens).most_common(top_n)
        words_s, counts_s = zip(*freq)

        ax.barh(list(words_s), list(counts_s), color=color, alpha=0.7)
        ax.set_title(f'Top {top_n} Words — {sentiment}')
        ax.set_xlabel('Frequency')
        ax.invert_yaxis()

    plt.suptitle('Top 10 Words by Sentiment', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("Top Words by Sentiment chart displayed ✅")

# ────────────────────────────────────────────────────────────────
# STEP 8 — Outlier Detection
# ────────────────────────────────────────────────────────────────
def plot_outlier_detection(data):
    numerical_cols = ['rating', 'helpful_votes', 'review_length']

    # ── 8.1 Box Plot ─────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=data[col], color='lightblue')
        plt.title(f'Boxplot — {col}')
        plt.ylabel(col)
    plt.suptitle('Outlier Detection — Box Plots', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ── 8.2 IQR Method ───────────────────────────────────────────
    print("\n--- Outlier Detection (IQR Method) ---")
    for col in numerical_cols:
        Q1  = data[col].quantile(0.25)
        Q3  = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

        print(f"\nColumn       : {col}")
        print(f"Q1           : {Q1}")
        print(f"Q3           : {Q3}")
        print(f"IQR          : {IQR}")
        print(f"Lower Bound  : {lower_bound}")
        print(f"Upper Bound  : {upper_bound}")
        print(f"Outlier Count: {len(outliers)}")

    # ── 8.3 Scatter Plot ─────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    for i, col in enumerate(numerical_cols, 1):
        Q1  = data[col].quantile(0.25)
        Q3  = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound  = Q1 - 1.5 * IQR
        upper_bound  = Q3 + 1.5 * IQR
        outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

        plt.subplot(1, 3, i)
        plt.scatter(range(len(data)), data[col],
                    c=outlier_mask.map({True: 'red', False: 'steelblue'}),
                    alpha=0.6, s=20)
        plt.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
        plt.axhline(y=lower_bound, color='green',  linestyle='--', label='Lower Bound')
        plt.title(f'Scatter — {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.legend(fontsize=7)

    plt.suptitle('Outlier Visualization — Scatter Plots', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ── 8.4 Review Length Distribution ───────────────────────────
    plt.figure(figsize=(8, 5))
    sns.histplot(data['review_length'], bins=20, kde=True, color='steelblue')
    plt.title('Review Length Distribution with KDE')
    plt.xlabel('Review Length')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # ── 8.5 Outlier Summary ──────────────────────────────────────
    print("\n--- Outlier Summary ---")
    total_outliers = 0
    for col in numerical_cols:
        Q1  = data[col].quantile(0.25)
        Q3  = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers    = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        total_outliers += len(outliers)
        print(f"{col:20}: {len(outliers)} outliers "
              f"({'none' if len(outliers) == 0 else 'review needed'})")

    print(f"\nTotal Outliers Found : {total_outliers}")
    print("Outlier Detection charts displayed ✅")

# ────────────────────────────────────────────────────────────────
# MASTER FUNCTION — Run All EDA Steps
# ────────────────────────────────────────────────────────────────
def run_eda(data):
    """
    Call this single function from any pipeline to run
    the complete EDA — all charts and outlier detection.
    """
    print("\n" + "=" * 55)
    print("  Starting EDA Pipeline")
    print("=" * 55)

    # Add sentiment column
    data['sentiment'] = data['rating'].apply(map_sentiment)
    print("\nSentiment column created.")
    print("Sentiment Counts:")
    print(data['sentiment'].value_counts())

    print("\n--- Step 2: Sentiment Distribution ---")
    plot_sentiment_distribution(data)

    print("\n--- Step 3: Rating Distribution ---")
    plot_rating_distribution(data)

    print("\n--- Step 4: Word Cloud (All Reviews) ---")
    plot_wordcloud_all(data)

    print("\n--- Step 5: Word Cloud by Sentiment ---")
    plot_wordcloud_by_sentiment(data)

    print("\n--- Step 6: Top 20 Most Frequent Words ---")
    plot_top_words(data, top_n=20)

    print("\n--- Step 7: Top 10 Words by Sentiment ---")
    plot_top_words_by_sentiment(data, top_n=10)

    print("\n--- Step 8: Outlier Detection ---")
    plot_outlier_detection(data)

    print("\n" + "=" * 55)
    print("  EDA Pipeline Completed ✅")
    print("=" * 55)
    print(f"\nTotal Reviews      : {len(data)}")
    print("\nSentiment Breakdown:")
    print(data['sentiment'].value_counts())
    print("\nRating Breakdown:")
    print(data['rating'].value_counts().sort_index())

    return data   # returns data with sentiment column added

# ────────────────────────────────────────────────────────────────
# Run directly or import and call run_eda()
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data(file_path)
    data = clean_data(data)
    data = run_eda(data)