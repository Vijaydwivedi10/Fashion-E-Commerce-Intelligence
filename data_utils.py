# data_utils.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64

from nlp_utils import analyzer, keyword_extractor, summarizer
from config import DATA_PATH, PLATFORMS, CATEGORIES, PRODUCTS, SAMPLE_REVIEWS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -------------------------------
# Configuration for Aspect Analysis
# -------------------------------
ASPECTS = [
    "Size & Fit", "Material Quality", "Packaging", "Delivery", "Price", "Design"
]
aspect_keywords = {
    "Size & Fit": ["size", "fit", "small", "large", "tight", "loose"],
    "Material Quality": ["fabric", "material", "feels", "cheap", "premium", "soft"],
    "Packaging": ["packaging", "box", "wrapped", "damaged", "torn"],
    "Delivery": ["delivery", "late", "fast", "delay", "shipping"],
    "Price": ["price", "expensive", "cheap", "cost", "value"],
    "Design": ["design", "style", "look", "color", "pattern"]
}

# Mapping of clusters/themes to actionable recommendations
THEME_ACTIONS = {
    "Runs small": "Review product’s sizing chart; consider adding half-sizes or fit-adjustment guides.",
    "Poor stitching": "Improve quality control on seams & thread strength.",
    "Delivery damaged": "Use reinforced packaging & audit carrier handling standards.",
    # Additional mappings can be added here
}

# -------------------------------
# Core Data Functions
# -------------------------------

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
        # Ensure Product column exists
        if 'Product' not in df.columns:
            df['Product'] = 'Unknown'
    else:
        cols = [
            "Timestamp", "Platform", "Product", "User", "Category", "Rating",
            "Review", "Sentiment", "SentimentScore", "Summary", "Keywords"
        ]
        df = pd.DataFrame(columns=cols)
    return df


def save_data(df):
    df.to_csv(DATA_PATH, index=False)


def simulate_reviews(n=500):
    """
    Simulate n reviews across platforms, categories, products, with basic NLP processing.
    """
    users = [f"user_{i}" for i in range(1, 501)]
    sample_reviews = SAMPLE_REVIEWS
    records = []
    now = datetime.now()

    for _ in range(n):
        dt = now - timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24)
        )
        platform = np.random.choice(PLATFORMS)
        user = np.random.choice(users)
        cat = np.random.choice(CATEGORIES)
        product = np.random.choice(PRODUCTS)
        rating = np.random.randint(1, 6)
        review = np.random.choice(sample_reviews)

        # Sentiment
        vs = analyzer.polarity_scores(review)
        comp = vs["compound"]
        sentiment = (
            "Positive" if comp >= 0.05 else
            "Negative" if comp <= -0.05 else
            "Neutral"
        )

        # Summary
        parser = PlaintextParser.from_string(review, Tokenizer("en"))
        summary_sentences = summarizer(parser.document, 1)
        summary = " ".join(str(s) for s in summary_sentences)

        # Keywords
        kws = keyword_extractor.extract_keywords(review)
        keywords = ", ".join([kw[0] for kw in kws])

        records.append({
            "Timestamp": dt,
            "Platform": platform,
            "Product": product,
            "User": user,
            "Category": cat,
            "Rating": rating,
            "Review": review,
            "Sentiment": sentiment,
            "SentimentScore": round(comp, 3),
            "Summary": summary,
            "Keywords": keywords
        })

    return pd.DataFrame(records)


def analyze_data(df):
    """Generate aggregated insights for dashboard charts."""
    insights = {}

    # Overall sentiment distribution
    dist = df["Sentiment"].value_counts().reset_index()
    dist.columns = ["Sentiment", "Count"]
    insights["sentiment_dist"] = dist

    # Sentiment by platform
    insights["by_platform"] = (
        df.groupby(["Platform", "Sentiment"]).size()
          .reset_index(name="Count")
    )

    # Sentiment over time (daily average)
    daily = (
        df.set_index("Timestamp").resample("D")["SentimentScore"].mean()
          .reset_index()
    )
    insights["sentiment_over_time"] = daily

    # Top negative keywords
    neg = df[df["Sentiment"] == "Negative"]
    word_list = []
    for kws in neg["Keywords"]:
        word_list.extend([w.strip() for w in kws.split(",")])
    top_complaints = (
        pd.Series(word_list).value_counts().head(10)
          .reset_index()
    )
    top_complaints.columns = ["Keyword", "Frequency"]
    insights["top_complaints"] = top_complaints

    # Category performance
    cat_perf = (
        df.groupby("Category")["Rating"].agg(["mean", "count"]) 
          .reset_index()
    )
    insights["category_performance"] = cat_perf

    return insights


def generate_report(df, insights):
    """Generate CSV report and return download link."""
    report = df.copy()
    csv = report.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

# -------------------------------
# Advanced Analysis Functions
# -------------------------------

def extract_aspect_sentiments(review: str) -> dict:
    """
    Extract and average sentiment for each defined aspect within a review.
    """
    sentences = nltk.sent_tokenize(review)
    scores = {asp: [] for asp in ASPECTS}
    for sentence in sentences:
        comp = analyzer.polarity_scores(sentence)["compound"]
        low = sentence.lower()
        for asp, kws in aspect_keywords.items():
            if any(k in low for k in kws):
                scores[asp].append(comp)
    return {asp: np.mean(scores[asp]) if scores[asp] else 0.0 for asp in ASPECTS}


def compute_aspect_dataframe(df: pd.DataFrame, text_col: str = 'Review') -> pd.DataFrame:
    """
    Apply aspect-sentiment extraction to each review and append aspect columns.
    """
    aspect_data = df[text_col].apply(extract_aspect_sentiments).tolist()
    aspect_df = pd.DataFrame(aspect_data)
    return pd.concat([df.reset_index(drop=True), aspect_df], axis=1)


def cluster_negative_feedback(
    df: pd.DataFrame,
    text_col: str = 'Review',
    threshold: float = -0.05,
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Cluster reviews with sentiment below threshold into themes.
    """
    neg_df = df[df['SentimentScore'] < threshold].copy()
    if neg_df.empty:
        neg_df['ThemeCluster'] = []
        return neg_df
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(neg_df[text_col])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    neg_df['ThemeCluster'] = labels
    return neg_df


def summarize_clusters(
    neg_df: pd.DataFrame,
    text_col: str = 'Review',
    top_n: int = 3
) -> pd.DataFrame:
    """
    Summarize each negative cluster with frequency and sample complaints.
    """
    recs = []
    for cluster in sorted(neg_df['ThemeCluster'].unique()):
        texts = neg_df[neg_df['ThemeCluster'] == cluster][text_col].tolist()
        recs.append({
            'Cluster':      cluster,
            'Frequency':    len(texts),
            'Samples':      texts[:top_n]
        })
    return pd.DataFrame(recs)


def map_actions(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map each cluster to a predefined action.
    """
    cluster_df['Action'] = cluster_df['Cluster'].map(
        lambda c: THEME_ACTIONS.get(str(c), 'No action defined.')
    )
    return cluster_df


def generate_action_plan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: aspect extraction, negative clustering, summarization, action mapping.
    """
    df_aspects = compute_aspect_dataframe(df)
    neg = cluster_negative_feedback(df_aspects)
    summary = summarize_clusters(neg)
    actions = map_actions(summary)
    return actions
