import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Smartest Conversational Partner",
    page_icon  = "💬",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 2.5rem; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .lbl {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 3px solid #a78bfa;
        padding-left: 10px;
        margin: 1.5rem 0 1rem;
    }
    .sentiment-badge-Positive {
        background: #064e3b; color: #34d399;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 1rem;
        display: inline-block;
    }
    .sentiment-badge-Negative {
        background: #450a0a; color: #f87171;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 1rem;
        display: inline-block;
    }
    .sentiment-badge-Neutral {
        background: #1e3a5f; color: #60a5fa;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 1rem;
        display: inline-block;
    }
    .stTextArea textarea {
        background: #1a1d2e !important;
        border: 1px solid #2d3148 !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-testid="stSidebar"] {
        background: #13151f;
        border-right: 1px solid #2d3148;
    }
    .stSelectbox > div > div {
        background: #1a1d2e !important;
        border: 1px solid #2d3148 !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Data\cleaned_data.csv"
PKL_DIR    = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\pklfiles\PKL_Outputs"
MODEL_DIR  = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\Model"

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    def map_sentiment(r):
        if r >= 4: return 'Positive'
        elif r == 3: return 'Neutral'
        else: return 'Negative'
    df['sentiment'] = df['rating'].apply(map_sentiment)
    return df

@st.cache_resource
def load_models():
    with open(f"{PKL_DIR}/label_encoder.pkl",    'rb') as f: le           = pickle.load(f)
    with open(f"{PKL_DIR}/tfidf_vectorizer.pkl", 'rb') as f: tfidf        = pickle.load(f)
    with open(f"{PKL_DIR}/scaler_tfidf.pkl",     'rb') as f: scaler_tfidf = pickle.load(f)
    with open(f"{MODEL_DIR}/onehot_encoder.pkl",         'rb') as f: ohe         = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder_version.pkl",  'rb') as f: le_version  = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder_verified.pkl", 'rb') as f: le_verified = pickle.load(f)

    # ✅ Correct way to load LR model
    with open(f"{MODEL_DIR}/lr_tfidf_model.pkl", 'rb') as f: model = pickle.load(f)

    return le, tfidf, scaler_tfidf, ohe, le_version, le_verified, model

df = load_data()
le, tfidf, scaler_tfidf, ohe, le_version, le_verified, best_model = load_models()

# ─────────────────────────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────────
def predict_sentiment(review, platform='Web', language='English',
                      location='Unknown', version='1.0.0',
                      verified_purchase='No', helpful_votes=0):

    review_length  = len(review.split())
    X_tfidf        = tfidf.transform([review]).toarray()
    X_tfidf_scaled = scaler_tfidf.transform(X_tfidf)

    cat_df   = pd.DataFrame([[platform, language, location]],
                              columns=['platform', 'language', 'location'])
    X_onehot = ohe.transform(cat_df)

    try:    v_enc  = le_version.transform([str(version)])[0]
    except: v_enc  = 0
    try:    vp_enc = le_verified.transform([str(verified_purchase)])[0]
    except: vp_enc = 0

    X_extra = np.hstack([X_onehot, [[v_enc]], [[vp_enc]], [[helpful_votes, review_length]]])
    X_full  = np.hstack([X_tfidf_scaled, X_extra])

    pred       = best_model.predict(X_full)[0]
    proba      = best_model.predict_proba(X_full)[0]
    confidence = round(float(np.max(proba)) * 100, 2)

    # ── Rule-based fallback when confidence is low ────────────────
    if confidence < 60:
        review_lower = review.lower()

        positive_words = ['good','great','amazing','excellent','love','perfect',
                          'best','awesome','fantastic','happy','satisfied',
                          'wonderful','helpful','easy','fast','smooth','nice',
                          'recommend','brilliant','superb','useful','friendly']

        negative_words = ['bad','terrible','horrible','worst','hate','awful',
                          'poor','slow','crash','bug','issue','problem','error',
                          'broken','useless','disappointed','frustrating','annoying',
                          'laggy','freeze','fail','waste','boring','difficult']

        pos_score = sum(1 for w in positive_words if w in review_lower)
        neg_score = sum(1 for w in negative_words if w in review_lower)

        if pos_score > neg_score:
            sentiment  = 'Positive'
            confidence = round(50 + (pos_score / (pos_score + neg_score + 1)) * 40, 2)
        elif neg_score > pos_score:
            sentiment  = 'Negative'
            confidence = round(50 + (neg_score / (pos_score + neg_score + 1)) * 40, 2)
        else:
            sentiment  = 'Neutral'
            confidence = 55.0

        scores = {
            'Positive' : confidence if sentiment == 'Positive' else round((100 - confidence) / 2, 2),
            'Neutral'  : confidence if sentiment == 'Neutral'  else round((100 - confidence) / 2, 2),
            'Negative' : confidence if sentiment == 'Negative' else round((100 - confidence) / 2, 2)
        }
        return sentiment, confidence, scores

    # ── Normal model prediction ───────────────────────────────────
    sentiment = le.inverse_transform([pred])[0]
    scores    = {le.classes_[i]: round(float(proba[i]) * 100, 2)
                 for i in range(len(le.classes_))}
    return sentiment, confidence, scores

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne;color:#a78bfa;'>💬 SCP</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af;font-size:0.8rem;'>Smartest Conversational Partner</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", [
    "🏠 Overview",
    "🔮 Predict Sentiment",
    "📊 EDA Dashboard",
    "🌍 Location Analysis",
    "📱 Platform & Version",
    "🧠 Sentiment Analysis Q&A",
    "🔁 Project Pipeline"
])
    st.divider()
    st.markdown(f"<p style='color:#6b7280;font-size:0.75rem;'>Dataset: {len(df)} reviews</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("<div class='hero-title'>Smartest Conversational Partner</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Sentiment Analysis Dashboard — Review Intelligence Platform</div>", unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total     = len(df)
    pos_pct   = round(len(df[df['sentiment'] == 'Positive']) / total * 100, 1)
    neg_pct   = round(len(df[df['sentiment'] == 'Negative']) / total * 100, 1)
    neu_pct   = round(len(df[df['sentiment'] == 'Neutral'])  / total * 100, 1)
    avg_r     = round(df['rating'].mean(), 2)

    with col1:
        st.markdown(f"<div class='metric-card'><div class='val'>{total}</div><div class='lbl'>Total Reviews</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='val' style='color:#34d399'>{pos_pct}%</div><div class='lbl'>Positive</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='val' style='color:#f87171'>{neg_pct}%</div><div class='lbl'>Negative</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='val' style='color:#60a5fa'>{neu_pct}%</div><div class='lbl'>Neutral</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><div class='val'>{avg_r}⭐</div><div class='lbl'>Avg Rating</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Rating Distribution</div>", unsafe_allow_html=True)
    import plotly.express as px
    import plotly.graph_objects as go

    col_a, col_b = st.columns(2)

    with col_a:
        rating_counts = df['rating'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        colors = ['#f87171','#fb923c','#facc15','#4ade80','#34d399']
        fig = px.bar(rating_counts, x='Rating', y='Count',
                     color='Rating',
                     color_discrete_sequence=colors,
                     title='⭐ Rating Distribution (1–5 Stars)')
        fig.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', showlegend=False,
            title_font_family='Syne'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sent_counts = df['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig2 = px.pie(sent_counts, names='Sentiment', values='Count',
                      color='Sentiment',
                      color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                      title='😊 Sentiment Distribution',
                      hole=0.45)
        fig2.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne'
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: PREDICT SENTIMENT
# ─────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Sentiment":
    import plotly.graph_objects as go

    st.markdown("<div class='hero-title'>🔮 Predict Sentiment</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Enter a review and get instant sentiment analysis</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='section-title'>Review Input</div>", unsafe_allow_html=True)
        review_text = st.text_area("Review Text", height=140,
                                    placeholder="Type or paste a review here...")

        c1, c2 = st.columns(2)
        with c1:
            platform   = st.selectbox("Platform",  df['platform'].unique().tolist())
            language   = st.selectbox("Language",  df['language'].unique().tolist())
            location   = st.selectbox("Location",  df['location'].unique().tolist())
        with c2:
            version    = st.selectbox("Version",   df['version'].unique().tolist())
            verified   = st.selectbox("Verified?", ['Yes', 'No'])
            helpful    = st.slider("Helpful Votes", 0, 200, 10)

        predict_btn = st.button("🔮 Predict Sentiment")

    with col_right:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        if predict_btn and review_text.strip():
            sentiment, confidence, scores = predict_sentiment(
                review_text, platform, language, location,
                version, verified, helpful
            )
            emoji_map = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}
            emoji     = emoji_map.get(sentiment, '')

            st.markdown(f"""
                <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;padding:1.5rem;text-align:center;margin-bottom:1rem;'>
                    <div style='font-size:3rem;'>{emoji}</div>
                    <div class='sentiment-badge-{sentiment}'>{sentiment}</div>
                    <div style='color:#9ca3af;margin-top:0.8rem;font-size:0.9rem;'>Confidence</div>
                    <div style='font-family:Syne;font-size:2rem;font-weight:700;color:#a78bfa;'>{confidence}%</div>
                </div>
            """, unsafe_allow_html=True)

            # Confidence bar chart
            fig = go.Figure(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation='h',
                marker_color=['#34d399' if k == 'Positive' else '#f87171' if k == 'Negative' else '#60a5fa'
                              for k in scores.keys()],
                text=[f"{v}%" for v in scores.values()],
                textposition='outside'
            ))
            fig.update_layout(
                paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                font_color='#e2e8f0', height=200,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(range=[0, 110])
            )
            st.plotly_chart(fig, use_container_width=True)

        elif predict_btn:
            st.warning("Please enter a review text first.")
        else:
            st.markdown("""
                <div style='background:#1a1d2e;border:1px dashed #2d3148;border-radius:12px;
                            padding:3rem;text-align:center;color:#4b5563;'>
                    <div style='font-size:2.5rem;'>🔮</div>
                    <div style='margin-top:0.5rem;'>Enter a review and click Predict</div>
                </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("<div class='hero-title'>📊 EDA Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Exploratory Data Analysis — Deep dive into review patterns</div>", unsafe_allow_html=True)

    # Q2 — Helpful votes
    st.markdown("<div class='section-title'>👍 Helpful Votes Distribution</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    threshold = 10
    helpful_above = len(df[df['helpful_votes'] > threshold])
    helpful_below = len(df[df['helpful_votes'] <= threshold])

    with col1:
        fig = go.Figure(go.Pie(
            labels=[f'> {threshold} helpful votes', f'≤ {threshold} helpful votes'],
            values=[helpful_above, helpful_below],
            hole=0.5,
            marker_colors=['#a78bfa', '#2d3148']
        ))
        fig.update_layout(
            paper_bgcolor='#1a1d2e', font_color='#e2e8f0',
            title='Helpful Vote Threshold (>10)', title_font_family='Syne'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='helpful_votes', nbins=20,
                             color_discrete_sequence=['#a78bfa'],
                             title='Helpful Votes Histogram')
        fig2.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Q8 — Review length per rating
    st.markdown("<div class='section-title'>🔠 Review Length by Rating</div>", unsafe_allow_html=True)
    fig3 = px.box(df, x='rating', y='review_length',
                  color='rating',
                  color_discrete_sequence=['#f87171','#fb923c','#facc15','#4ade80','#34d399'],
                  title='Review Length Distribution per Star Rating')
    fig3.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', showlegend=False, title_font_family='Syne'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Q7 — Verified vs non-verified
    st.markdown("<div class='section-title'>✅ Verified vs Non-Verified Satisfaction</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    verified_avg = df.groupby('verified_purchase')['rating'].mean().reset_index()
    verified_avg.columns = ['Verified', 'Avg Rating']

    with col3:
        fig4 = px.bar(verified_avg, x='Verified', y='Avg Rating',
                      color='Verified',
                      color_discrete_map={'Yes':'#34d399','No':'#f87171'},
                      title='Avg Rating: Verified vs Non-Verified')
        fig4.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', showlegend=False, title_font_family='Syne',
            yaxis=dict(range=[0, 5])
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        verified_sent = df.groupby(['verified_purchase','sentiment']).size().reset_index(name='count')
        fig5 = px.bar(verified_sent, x='verified_purchase', y='count',
                      color='sentiment',
                      color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                      barmode='group',
                      title='Sentiment by Verified Purchase')
        fig5.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne'
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Q3 — Top keywords
    st.markdown("<div class='section-title'>🧭 Top Keywords: Positive vs Negative Reviews</div>", unsafe_allow_html=True)
    from collections import Counter
    import re

    def get_top_words(texts, n=15):
        words = []
        for t in texts:
            words.extend(re.findall(r'\b[a-z]{3,}\b', str(t).lower()))
        return Counter(words).most_common(n)

    pos_words = get_top_words(df[df['sentiment'] == 'Positive']['review'])
    neg_words = get_top_words(df[df['sentiment'] == 'Negative']['review'])

    col5, col6 = st.columns(2)
    with col5:
        pw_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
        fig6  = px.bar(pw_df, x='Count', y='Word', orientation='h',
                       color_discrete_sequence=['#34d399'],
                       title='😊 Top Words in Positive Reviews')
        fig6.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col6:
        nw_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
        fig7  = px.bar(nw_df, x='Count', y='Word', orientation='h',
                       color_discrete_sequence=['#f87171'],
                       title='😞 Top Words in Negative Reviews')
        fig7.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig7, use_container_width=True)

    # Q9 — 1-star specific words
    st.markdown("<div class='section-title'>💬 Most Mentioned Words in 1-Star Reviews</div>", unsafe_allow_html=True)
    one_star_words = get_top_words(df[df['rating'] == 1]['review'], n=20)
    osw_df = pd.DataFrame(one_star_words, columns=['Word', 'Count'])
    fig8   = px.bar(osw_df, x='Word', y='Count',
                    color='Count',
                    color_continuous_scale=['#450a0a','#f87171'],
                    title='Top Terms in 1-Star Reviews')
    fig8.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', title_font_family='Syne', showlegend=False
    )
    st.plotly_chart(fig8, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: LOCATION ANALYSIS
# ─────────────────────────────────────────────────────────────────
elif page == "🌍 Location Analysis":
    import plotly.express as px

    st.markdown("<div class='hero-title'>🌍 Location Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>How ratings and sentiment vary by user location</div>", unsafe_allow_html=True)

    # Q5 — Ratings by location
    st.markdown("<div class='section-title'>📍 Average Rating by Location</div>", unsafe_allow_html=True)
    loc_avg = df.groupby('location')['rating'].mean().reset_index()
    loc_avg.columns = ['Location', 'Avg Rating']
    loc_avg = loc_avg.sort_values('Avg Rating', ascending=False)

    fig = px.bar(loc_avg, x='Location', y='Avg Rating',
                 color='Avg Rating',
                 color_continuous_scale=['#f87171','#facc15','#34d399'],
                 title='Average Rating by Location')
    fig.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', title_font_family='Syne',
        yaxis=dict(range=[0, 5])
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment breakdown by location
    st.markdown("<div class='section-title'>😊 Sentiment Breakdown by Location</div>", unsafe_allow_html=True)
    loc_sent = df.groupby(['location','sentiment']).size().reset_index(name='count')
    fig2 = px.bar(loc_sent, x='location', y='count',
                  color='sentiment',
                  color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                  barmode='stack',
                  title='Sentiment Distribution by Location')
    fig2.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', title_font_family='Syne'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Top & Bottom locations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>🏆 Top 5 Locations</div>", unsafe_allow_html=True)
        st.dataframe(
            loc_avg.head(5).style.background_gradient(cmap='Greens'),
            use_container_width=True
        )
    with col2:
        st.markdown("<div class='section-title'>⚠️ Bottom 5 Locations</div>", unsafe_allow_html=True)
        st.dataframe(
            loc_avg.tail(5).style.background_gradient(cmap='Reds'),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────
# PAGE: PLATFORM & VERSION
# ─────────────────────────────────────────────────────────────────
elif page == "📱 Platform & Version":
    import plotly.express as px

    st.markdown("<div class='hero-title'>📱 Platform & Version</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Platform performance and version-wise rating trends</div>", unsafe_allow_html=True)

    # Q6 — Platform comparison
    st.markdown("<div class='section-title'>🖥️ Platform Rating Comparison</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    plat_avg  = df.groupby('platform')['rating'].mean().reset_index()
    plat_avg.columns = ['Platform', 'Avg Rating']
    plat_sent = df.groupby(['platform','sentiment']).size().reset_index(name='count')

    with col1:
        fig = px.bar(plat_avg, x='Platform', y='Avg Rating',
                     color='Platform',
                     color_discrete_sequence=['#a78bfa','#60a5fa','#34d399','#fb923c'],
                     title='Average Rating by Platform')
        fig.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne',
            showlegend=False, yaxis=dict(range=[0, 5])
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(plat_sent, x='platform', y='count',
                      color='sentiment',
                      color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                      barmode='group',
                      title='Sentiment by Platform')
        fig2.update_layout(
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font_color='#e2e8f0', title_font_family='Syne'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Q10 — Version rating
    st.markdown("<div class='section-title'>🔢 Average Rating by Version</div>", unsafe_allow_html=True)
    ver_avg = df.groupby('version')['rating'].mean().reset_index()
    ver_avg.columns = ['Version', 'Avg Rating']
    ver_avg = ver_avg.sort_values('Version')

    fig3 = px.bar(ver_avg, x='Version', y='Avg Rating',
                  color='Avg Rating',
                  color_continuous_scale=['#f87171','#facc15','#34d399'],
                  text='Avg Rating',
                  title='Which Version Got the Best Ratings?')
    fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig3.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', title_font_family='Syne',
        yaxis=dict(range=[0, 5.5])
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Language breakdown
    st.markdown("<div class='section-title'>🌐 Rating by Language</div>", unsafe_allow_html=True)
    lang_avg = df.groupby('language')['rating'].mean().reset_index()
    lang_avg.columns = ['Language', 'Avg Rating']
    lang_avg = lang_avg.sort_values('Avg Rating', ascending=False)

    fig4 = px.bar(lang_avg, x='Language', y='Avg Rating',
                  color='Avg Rating',
                  color_continuous_scale=['#f87171','#facc15','#34d399'],
                  title='Average Rating by Language')
    fig4.update_layout(
        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
        font_color='#e2e8f0', title_font_family='Syne',
        yaxis=dict(range=[0, 5])
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: SENTIMENT ANALYSIS Q&A
# ─────────────────────────────────────────────────────────────────
elif page == "🧠 Sentiment Analysis Q&A":
    import plotly.express as px
    import plotly.graph_objects as go
    from collections import Counter
    import re

    st.markdown("<div class='hero-title'>🧠 Sentiment Analysis Q&A</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>10 Key Questions answered with data-driven visualizations</div>", unsafe_allow_html=True)

    def get_top_words(texts, n=15):
        stopwords = {'the','and','is','in','it','of','to','a','was','for',
                     'on','are','with','as','this','that','be','at','by',
                     'an','we','not','app','use','very','have','been','but'}
        words = []
        for t in texts:
            words.extend([w for w in re.findall(r'\b[a-z]{3,}\b', str(t).lower())
                          if w not in stopwords])
        return Counter(words).most_common(n)

    # ── Q1: Overall Sentiment ─────────────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q1.</span>
        <span style='color:#e2e8f0;font-weight:500;'> What is the overall sentiment of user reviews?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Classify each review as Positive, Neutral, or Negative and compute proportions.
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    sent_counts = df['sentiment'].value_counts().reset_index()
    sent_counts.columns = ['Sentiment', 'Count']
    with col1:
        fig = px.pie(sent_counts, names='Sentiment', values='Count', hole=0.5,
                     color='Sentiment',
                     color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'})
        fig.update_layout(paper_bgcolor='#1a1d2e', font_color='#e2e8f0',
                          title='Overall Sentiment Distribution', title_font_family='Syne')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        total = len(df)
        for s, color in [('Positive','#34d399'),('Neutral','#60a5fa'),('Negative','#f87171')]:
            cnt = len(df[df['sentiment'] == s])
            pct = round(cnt / total * 100, 1)
            st.markdown(f"""
            <div style='background:#1a1d2e;border-left:4px solid {color};border-radius:8px;
                        padding:0.8rem 1.2rem;margin-bottom:0.8rem;display:flex;
                        justify-content:space-between;align-items:center;'>
                <span style='color:#e2e8f0;font-weight:500;'>{s}</span>
                <span style='color:{color};font-family:Syne;font-size:1.4rem;font-weight:700;'>{pct}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6b7280;font-size:0.8rem;margin-top:0.5rem;'>Based on {total} reviews</div>",
                    unsafe_allow_html=True)

    # ── Q2: Sentiment vs Rating mismatch ─────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q2.</span>
        <span style='color:#e2e8f0;font-weight:500;'> How does sentiment vary by rating?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Do 1-star reviews always contain negative sentiment? Any mismatch between rating and text?
        </div>
    </div>""", unsafe_allow_html=True)

    heat_data = df.groupby(['rating','sentiment']).size().reset_index(name='count')
    fig2 = px.bar(heat_data, x='rating', y='count', color='sentiment',
                  barmode='stack',
                  color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                  title='Sentiment Distribution per Star Rating',
                  labels={'rating':'Star Rating','count':'Number of Reviews'})
    fig2.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                       font_color='#e2e8f0', title_font_family='Syne')
    st.plotly_chart(fig2, use_container_width=True)

    # Mismatch table
    mismatch = df[((df['rating'] >= 4) & (df['sentiment'] == 'Negative')) |
                  ((df['rating'] <= 2) & (df['sentiment'] == 'Positive'))]
    st.markdown(f"<div style='color:#facc15;font-size:0.9rem;margin-bottom:0.5rem;'>⚠️ Sentiment-Rating mismatches found: <b>{len(mismatch)}</b> reviews</div>",
                unsafe_allow_html=True)

    # ── Q3: Keywords per sentiment ────────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q3.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Which keywords are most associated with each sentiment?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Keyword frequency tables per sentiment type.
        </div>
    </div>""", unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    for col, sentiment, color in zip([col3,col4,col5],
                                      ['Positive','Neutral','Negative'],
                                      ['#34d399','#60a5fa','#f87171']):
        words = get_top_words(df[df['sentiment'] == sentiment]['review'], n=10)
        wdf   = pd.DataFrame(words, columns=['Keyword','Count'])
        fig   = px.bar(wdf, x='Count', y='Keyword', orientation='h',
                       color_discrete_sequence=[color],
                       title=f'{sentiment} Keywords')
        fig.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                          font_color='#e2e8f0', title_font_family='Syne',
                          margin=dict(l=5,r=5,t=40,b=5), height=300,
                          yaxis={'categoryorder':'total ascending'})
        with col:
            st.plotly_chart(fig, use_container_width=True)

    # ── Q4: Sentiment over time (simulated) ──────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q4.</span>
        <span style='color:#e2e8f0;font-weight:500;'> How has sentiment changed over time?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Sentiment trends by simulated month index (no date column in dataset).
        </div>
    </div>""", unsafe_allow_html=True)

    df_time       = df.copy()
    df_time['month_idx'] = (df_time.index // 50) + 1
    time_sent     = df_time.groupby(['month_idx','sentiment']).size().reset_index(name='count')
    fig_time      = px.line(time_sent, x='month_idx', y='count', color='sentiment',
                            color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                            markers=True,
                            title='Sentiment Trend Over Time (Batch Index)',
                            labels={'month_idx':'Batch (50 reviews each)','count':'Review Count'})
    fig_time.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                           font_color='#e2e8f0', title_font_family='Syne')
    st.plotly_chart(fig_time, use_container_width=True)
    st.markdown("<div style='color:#6b7280;font-size:0.8rem;'>ℹ️ Date column not available — data split into batches of 50 to simulate time trend.</div>",
                unsafe_allow_html=True)

    # ── Q5: Verified vs Non-verified ─────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q5.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Do verified users leave more positive or negative reviews?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Sentiment distribution — verified_purchase Yes vs No.
        </div>
    </div>""", unsafe_allow_html=True)

    col6, col7 = st.columns(2)
    ver_sent   = df.groupby(['verified_purchase','sentiment']).size().reset_index(name='count')
    with col6:
        fig_vs = px.bar(ver_sent, x='verified_purchase', y='count', color='sentiment',
                        barmode='group',
                        color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                        title='Sentiment by Verified Purchase')
        fig_vs.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                             font_color='#e2e8f0', title_font_family='Syne')
        st.plotly_chart(fig_vs, use_container_width=True)
    with col7:
        ver_pos_pct = df.groupby('verified_purchase').apply(
            lambda x: round((x['sentiment'] == 'Positive').sum() / len(x) * 100, 1)
        ).reset_index()
        ver_pos_pct.columns = ['Verified', 'Positive %']
        fig_vp = px.bar(ver_pos_pct, x='Verified', y='Positive %',
                        color='Verified',
                        color_discrete_map={'Yes':'#34d399','No':'#f87171'},
                        title='% Positive Reviews by Verified Status')
        fig_vp.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                             font_color='#e2e8f0', title_font_family='Syne',
                             showlegend=False, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_vp, use_container_width=True)

    # ── Q6: Review length vs sentiment ───────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q6.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Are longer reviews more likely to be negative or positive?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Average review length compared across sentiment classes.
        </div>
    </div>""", unsafe_allow_html=True)

    col8, col9 = st.columns(2)
    len_sent   = df.groupby('sentiment')['review_length'].mean().reset_index()
    len_sent.columns = ['Sentiment','Avg Length']
    with col8:
        fig_len = px.bar(len_sent, x='Sentiment', y='Avg Length',
                         color='Sentiment',
                         color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                         title='Avg Review Length per Sentiment')
        fig_len.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                              font_color='#e2e8f0', title_font_family='Syne', showlegend=False)
        st.plotly_chart(fig_len, use_container_width=True)
    with col9:
        fig_box = px.box(df, x='sentiment', y='review_length', color='sentiment',
                         color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                         title='Review Length Distribution per Sentiment')
        fig_box.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                              font_color='#e2e8f0', title_font_family='Syne', showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Q7: Location sentiment ────────────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q7.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Which locations show the most positive or negative sentiment?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Region-based user experience issues or appreciation.
        </div>
    </div>""", unsafe_allow_html=True)

    loc_pos = df[df['sentiment']=='Positive'].groupby('location').size()
    loc_neg = df[df['sentiment']=='Negative'].groupby('location').size()
    loc_df  = pd.DataFrame({'Positive': loc_pos, 'Negative': loc_neg}).fillna(0).reset_index()
    loc_df['Net Score'] = loc_df['Positive'] - loc_df['Negative']
    loc_df  = loc_df.sort_values('Net Score', ascending=False)

    fig_loc = px.bar(loc_df, x='location', y='Net Score',
                     color='Net Score',
                     color_continuous_scale=['#f87171','#6b7280','#34d399'],
                     title='Net Sentiment Score by Location (Positive − Negative)')
    fig_loc.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                          font_color='#e2e8f0', title_font_family='Syne')
    st.plotly_chart(fig_loc, use_container_width=True)

    # ── Q8: Platform sentiment ────────────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q8.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Is there a difference in sentiment across platforms?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Identify where user experience needs improvement — Web vs Mobile.
        </div>
    </div>""", unsafe_allow_html=True)

    plat_sent = df.groupby(['platform','sentiment']).size().reset_index(name='count')
    plat_pct  = df.groupby('platform').apply(
        lambda x: round((x['sentiment'] == 'Positive').sum() / len(x) * 100, 1)
    ).reset_index()
    plat_pct.columns = ['Platform', 'Positive %']

    col10, col11 = st.columns(2)
    with col10:
        fig_plat = px.bar(plat_sent, x='platform', y='count', color='sentiment',
                          barmode='stack',
                          color_discrete_map={'Positive':'#34d399','Neutral':'#60a5fa','Negative':'#f87171'},
                          title='Sentiment Stack by Platform')
        fig_plat.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                               font_color='#e2e8f0', title_font_family='Syne')
        st.plotly_chart(fig_plat, use_container_width=True)
    with col11:
        fig_pp = px.bar(plat_pct, x='Platform', y='Positive %',
                        color='Platform',
                        color_discrete_sequence=['#a78bfa','#60a5fa','#34d399','#fb923c'],
                        title='% Positive Reviews per Platform')
        fig_pp.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                             font_color='#e2e8f0', title_font_family='Syne',
                             showlegend=False, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_pp, use_container_width=True)

    # ── Q9: Version sentiment ─────────────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q9.</span>
        <span style='color:#e2e8f0;font-weight:500;'> Which ChatGPT versions are associated with higher/lower sentiment?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Determine if a version release impacted user satisfaction.
        </div>
    </div>""", unsafe_allow_html=True)

    ver_sent_pct = df.groupby('version').apply(
        lambda x: pd.Series({
            'Positive %': round((x['sentiment']=='Positive').sum()/len(x)*100,1),
            'Negative %': round((x['sentiment']=='Negative').sum()/len(x)*100,1),
            'Neutral %' : round((x['sentiment']=='Neutral').sum()/len(x)*100,1),
            'Avg Rating': round(x['rating'].mean(), 2)
        })
    ).reset_index().sort_values('version')

    fig_ver = go.Figure()
    fig_ver.add_trace(go.Bar(name='Positive %', x=ver_sent_pct['version'],
                             y=ver_sent_pct['Positive %'], marker_color='#34d399'))
    fig_ver.add_trace(go.Bar(name='Neutral %',  x=ver_sent_pct['version'],
                             y=ver_sent_pct['Neutral %'],  marker_color='#60a5fa'))
    fig_ver.add_trace(go.Bar(name='Negative %', x=ver_sent_pct['version'],
                             y=ver_sent_pct['Negative %'], marker_color='#f87171'))
    fig_ver.update_layout(barmode='group',
                          paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                          font_color='#e2e8f0', title='Sentiment % by Version',
                          title_font_family='Syne')
    st.plotly_chart(fig_ver, use_container_width=True)

    # ── Q10: Negative feedback themes ────────────────────────────
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
                padding:1rem 1.5rem;margin:1.5rem 0 0.5rem;'>
        <span style='color:#a78bfa;font-family:Syne;font-weight:700;'>Q10.</span>
        <span style='color:#e2e8f0;font-weight:500;'> What are the most common negative feedback themes?</span>
        <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px;'>
        Keyword grouping to identify recurring pain points in negative reviews.
        </div>
    </div>""", unsafe_allow_html=True)

    neg_reviews = df[df['sentiment'] == 'Negative']['review']
    neg_words   = get_top_words(neg_reviews, n=20)
    nw_df       = pd.DataFrame(neg_words, columns=['Theme','Count'])

    col12, col13 = st.columns([1.5, 1])
    with col12:
        fig_neg = px.bar(nw_df, x='Count', y='Theme', orientation='h',
                         color='Count',
                         color_continuous_scale=['#450a0a','#f87171'],
                         title='Top Negative Feedback Themes')
        fig_neg.update_layout(paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                              font_color='#e2e8f0', title_font_family='Syne',
                              yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_neg, use_container_width=True)
    with col13:
        st.markdown("<div class='section-title'>🔥 Top Pain Points</div>", unsafe_allow_html=True)
        for i, (word, count) in enumerate(neg_words[:8], 1):
            intensity = min(int(count / max([c for _,c in neg_words]) * 10), 10)
            bar       = '▓' * intensity + '░' * (10 - intensity)
            st.markdown(f"""
            <div style='background:#1a1d2e;border:1px solid #2d3148;border-radius:8px;
                        padding:0.5rem 1rem;margin-bottom:0.4rem;display:flex;
                        justify-content:space-between;align-items:center;'>
                <span style='color:#f87171;font-weight:500;'>#{i} {word}</span>
                <span style='color:#6b7280;font-size:0.75rem;letter-spacing:1px;'>{bar} {count}</span>
            </div>""", unsafe_allow_html=True)

    # Summary insight box
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e1b4b,#1a1d2e);
                border:1px solid #4c1d95;border-radius:12px;
                padding:1.5rem;margin-top:2rem;'>
        <div style='font-family:Syne;color:#a78bfa;font-weight:700;font-size:1.1rem;margin-bottom:0.8rem;'>
            📋 Key Insights Summary
        </div>
        <ul style='color:#d1d5db;line-height:2;margin:0;padding-left:1.2rem;'>
            <li>Sentiment is directly derived from ratings — higher star = more positive</li>
            <li>Verified users and non-verified users show similar sentiment patterns</li>
            <li>Negative reviews tend to use words like "bug", "issue", "crash", "poor"</li>
            <li>Positive reviews use words like "amazing", "great", "easy", "satisfied"</li>
            <li>Review length is consistent across sentiment classes in this dataset</li>
        </ul>
    </div>""", unsafe_allow_html=True)
    
elif page == "🔁 Project Pipeline":
    st.markdown("<div class='hero-title'>🔁 Project Pipeline</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>End-to-end ML pipeline architecture</div>",
                unsafe_allow_html=True)

    IMG_PATH = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner\App\SCP_Pipeline.png"

    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, width='stretch')
    else:
        st.warning("Image not found!")

    st.markdown("<div class='section-title'>Pipeline Summary</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1a1d2e;border:1px solid #2d3148;
                    border-radius:12px;padding:1rem 1.5rem;'>
            <p style='color:#a78bfa;font-weight:600;margin-bottom:0.5rem;'>Stages</p>
            <ul style='color:#d1d5db;line-height:2;margin:0;padding-left:1.2rem;'>
                <li>Data Loading → loader.py</li>
                <li>Data Cleaning → Data_Cleaning.py</li>
                <li>EDA → eda.py</li>
                <li>Feature Extraction → Featureextraction_1.py</li>
                <li>Model Training → Model_train.py</li>
                <li>Prediction → Step3_Prediction.py</li>
                <li>Dashboard → app.py</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1a1d2e;border:1px solid #2d3148;
                    border-radius:12px;padding:1rem 1.5rem;'>
            <p style='color:#a78bfa;font-weight:600;margin-bottom:0.5rem;'>Models Trained</p>
            <ul style='color:#d1d5db;line-height:2;margin:0;padding-left:1.2rem;'>
                <li>LR + TF-IDF ✅ Best model</li>
                <li>LR + BERT</li>
                <li>RF + TF-IDF</li>
                <li>RF + BERT</li>
                <li>LSTM + BERT</li>
            </ul>
        </div>""", unsafe_allow_html=True)

   