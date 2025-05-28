# review_insights.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_aspect_sentiments(df):
    """
    Plot average sentiment for each aspect.
    """
    aspect_cols = [col for col in df.columns if col in [
        "Size & Fit", "Material Quality", "Packaging", "Delivery", "Price", "Design"]]
    aspect_means = df[aspect_cols].mean().sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=aspect_means.index, y=aspect_means.values, palette="coolwarm")
    plt.title("Average Sentiment by Aspect")
    plt.ylabel("Sentiment Score")
    plt.xlabel("Aspect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def most_common_negative_aspects(df):
    """
    Find aspects with most negative mentions and count of negative reviews.
    """
    aspect_cols = [col for col in df.columns if col in [
        "Size & Fit", "Material Quality", "Packaging", "Delivery", "Price", "Design"]]
    negative_counts = {aspect: (df[aspect] < -0.2).sum() for aspect in aspect_cols}
    return pd.DataFrame(list(negative_counts.items()), columns=['Aspect', 'Negative Review Count']).sort_values(by="Negative Review Count", ascending=False)


def plot_negative_aspect_distribution(df):
    """
    Visualize the count of negative mentions across aspects.
    """
    neg_df = most_common_negative_aspects(df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Aspect', y='Negative Review Count', data=neg_df, palette="Reds_r")
    plt.title("Negative Review Counts by Aspect")
    plt.ylabel("Count")
    plt.xlabel("Aspect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_seller_insights(df):
    """
    Print insightful summary for sellers from aspect sentiment and frequency.
    """
    insights = []
    for aspect in ["Size & Fit", "Material Quality", "Packaging", "Delivery", "Price", "Design"]:
        mean_score = df[aspect].mean()
        neg_count = (df[aspect] < -0.2).sum()
        if mean_score < 0.1:
            insights.append(f"\n⚠️ *{aspect}* has low average sentiment ({mean_score:.2f}) and {neg_count} negative reviews. Consider reviewing this area.")
        elif neg_count > 0:
            insights.append(f"\n📌 *{aspect}* has some negativity ({neg_count} mentions). Could be improved further despite acceptable sentiment ({mean_score:.2f}).")
    return "\n".join(insights) if insights else "All aspects are performing well."
