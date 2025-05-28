import os
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from groq import Groq





# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Aspect definitions (used externally in app.py)
ASPECTS = ["Size & Fit", "Material Quality", "Packaging", "Delivery", "Price", "Design"]
aspect_keywords: Dict[str, List[str]] = {
    "Size & Fit": ["size", "fit", "small", "large", "tight", "loose", "snug", "oversized"],
    "Material Quality": ["fabric", "material", "feels", "cheap", "premium", "soft", "durable", "sturdy"],
    "Packaging": ["packaging", "box", "wrapped", "damaged", "torn", "intact"],
    "Delivery": ["delivery", "late", "fast", "delay", "shipping", "on time"],
    "Price": ["price", "expensive", "cheap", "cost", "value", "worth"],
    "Design": ["design", "style", "look", "color", "pattern", "aesthetic"]
}

# Short actions mapping (used externally)
SHORT_ACTIONS: Dict[str, str] = {
    "material": "Improve material quality.",
    "delivery": "Enhance delivery process.",
    "packaging": "Upgrade packaging.",
    "fit": "Refine sizing guide.",
    "price": "Adjust pricing strategy.",
    "design": "Iterate on design."
}

class FeedbackAnalyzer:
    def __init__(self, llm_api_key: str = None, model_name: str = "llama3-70b-8192"):
        """
        Initializes the Sentiment and LLM clients.
        llm_api_key: Groq API key or None to use GROQ_API_KEY env var
        model_name: Model identifier for Groq chat
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.model_name = model_name
        api_key = llm_api_key or os.getenv('GROQ_API_KEY', '')
        self.llm = Groq(api_key="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")

    def extract_aspect_sentiments(self, review: str) -> Dict[str, float]:
        sentences = sent_tokenize(review)
        scores = {asp: [] for asp in ASPECTS}
        for sent in sentences:
            comp = self.sentiment_analyzer.polarity_scores(sent)['compound']
            low = sent.lower()
            for aspect, keywords in aspect_keywords.items():
                if any(kw in low for kw in keywords):
                    scores[aspect].append(comp)
        return {asp: float(np.mean(vals)) if vals else 0.0 for asp, vals in scores.items()}

    def compute_aspect_dataframe(self, df: pd.DataFrame, text_col: str = 'Review') -> pd.DataFrame:
        logger.info("Computing aspect sentiments for %d reviews", len(df))
        matrix = df[text_col].apply(self.extract_aspect_sentiments).tolist()
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(matrix)], axis=1)

    def compute_overall_sentiment(self, df: pd.DataFrame, text_col: str = 'Review') -> pd.DataFrame:
        df['SentimentScore'] = df[text_col].apply(lambda x: self.sentiment_analyzer.polarity_scores(x)['compound'])
        return df

    def cluster_negative_feedback(
        self, df: pd.DataFrame, text_col: str = 'Review', neg_threshold: float = -0.05, max_clusters: int = 10
    ) -> pd.DataFrame:
        neg = df[df['SentimentScore'] < neg_threshold].copy()
        if neg.empty:
            return pd.DataFrame()
        vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
        X = vec.fit_transform(neg[text_col])
        best_k, best_score = 2, -1
        for k in range(2, min(max_clusters, len(neg)) + 1):
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        logger.info("Chosen %d clusters for negative feedback (silhouette=%.3f)", best_k, best_score)
        neg['ThemeCluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(X)
        return neg

    def summarize_clusters(self, neg_df: pd.DataFrame, text_col: str = 'Review') -> pd.DataFrame:
        summaries = []
        stops = set(nltk.corpus.stopwords.words('english'))
        for cid in sorted(neg_df['ThemeCluster'].unique()):
            subset = neg_df[neg_df['ThemeCluster'] == cid]
            tokens = [
                w for text in subset[text_col]
                for w in nltk.word_tokenize(text.lower())
                if w.isalpha() and w not in stops
            ]
            top = pd.Series(tokens).value_counts().head(5).index.tolist()
            theme = " ".join(top) or f"cluster_{cid}"
            samples = subset[text_col].drop_duplicates().tolist()[:3]
            summaries.append({'Cluster': theme, 'Frequency': len(subset), 'Samples': samples})
        return pd.DataFrame(summaries)

    def generate_unique_detailed_action(self, theme: str, samples: List[str]) -> str:
        """
        Use Groq chat completion to generate a unique, creative action plan.
        """
        prompt = (
            f"You are an expert e-commerce consultant. Propose a detailed, unique action plan to address the feedback theme: '{theme}'. "
            f"Complaints: {samples}. Ensure it does not duplicate other clusters. also detailed plan should start with 'Detailed-plan: Then give here the detailed plan' means directly write detailed plan don't copy from prompt anything like Here is a detailed don't write these things write plan directlty."
        )
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=120,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def map_actions(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        used, shorts, details = set(), [], []
        for row in summary_df.itertuples():
            th = row.Cluster.lower()
            short = next((v for k, v in SHORT_ACTIONS.items() if k in th), "Review feedback manually.")
            shorts.append(short)
            detail = self.generate_unique_detailed_action(row.Cluster, row.Samples)
            if detail in used:
                detail += " Focus on additional operational nuances to differentiate."
            used.add(detail)
            details.append(detail)
        summary_df['ShortAction'] = shorts
        summary_df['DetailedAction'] = details
        return summary_df

    def generate_action_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_overall_sentiment(df)
        df = self.compute_aspect_dataframe(df)
        neg = self.cluster_negative_feedback(df)
        if neg.empty:
            return pd.DataFrame([{
                'Cluster': 'No negative feedback',
                'Frequency': 0,
                'Samples': [],
                'ShortAction': 'No action needed.',
                'DetailedAction': 'All feedback is positive.'
            }])
        summary = self.summarize_clusters(neg)
        return self.map_actions(summary)

# Default instance using env var
_feedback_analyzer = FeedbackAnalyzer()

# Module-level functions for app.py imports
def compute_aspect_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _feedback_analyzer.compute_aspect_dataframe(df)

def cluster_negative_feedback(df: pd.DataFrame) -> pd.DataFrame:
    return _feedback_analyzer.cluster_negative_feedback(df)

def summarize_clusters(neg_df: pd.DataFrame) -> pd.DataFrame:
    return _feedback_analyzer.summarize_clusters(neg_df)

def map_actions(summary_df: pd.DataFrame) -> pd.DataFrame:
    return _feedback_analyzer.map_actions(summary_df)

def generate_action_plan(df: pd.DataFrame) -> pd.DataFrame:
    return _feedback_analyzer.generate_action_plan(df)
