import streamlit as st
import pandas as pd
from groq import Groq

# External dependencies
from data_utils import load_data
from analysis_utils import (
    compute_aspect_dataframe,
    cluster_negative_feedback,
    summarize_clusters,
    map_actions
)
from config import PLATFORMS, CATEGORIES, PRODUCTS




# Initialize Groq client with your API key
client = Groq(api_key="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")

# Recommended replacement model since mixtral-8x7b is deprecated
MODEL_NAME = "llama-3.3-70b-versatile"



def deep_product_insights(df: pd.DataFrame):
    st.markdown("---")
    st.subheader("🔍 Deep Product Analysis")

    # Selection filters
    col1, col2, col3 = st.columns(3)
    sel_cat = col1.selectbox("Select Category", [None] + CATEGORIES)
    sel_brand = col2.selectbox("Select Brand", [None] + PLATFORMS)
    sel_prod = col3.selectbox("Select Product", [None] + PRODUCTS)

    analyze = st.button(
        "Analyze Product",
        key="analyze_prod",
        help="Analyze a specific product",
        use_container_width=True
    )
    if not analyze:
        return

    # Validation
    missing = [name for name, val in [
        ('Category', sel_cat),
        ('Brand', sel_brand),
        ('Product', sel_prod)] if val is None]
    if missing:
        st.error(f"{', '.join(missing)} cannot be None. Please select all fields.")
        return

    # Filter data
    filtered = df[
        (df['Category'] == sel_cat) &
        (df['Platform'] == sel_brand) &
        (df['Product'] == sel_prod)
    ]
    if filtered.empty:
        st.warning("No reviews found for this selection.")
        return

    # Aggregate metrics
    avg_rating = filtered['Rating'].mean()
    sentiment_counts = filtered['Sentiment'].value_counts().to_dict()
    top_keywords = filtered['Keywords'].str.split(', ').explode().value_counts().head(5).to_dict()

    # Prepare prompt
    prompt = (
        f"As an e-commerce analyst, provide an overview for the product '{sel_prod}' "
        f"in category '{sel_cat}' by brand '{sel_brand}'. It has an average rating of {avg_rating:.2f}, "
        f"sentiment distribution: {sentiment_counts}, top keywords: {top_keywords}. "
        f"Summarize what customers liked most and what they complained about, in 3 concise bullet points."
    )
    # LLM call with updated model
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    overview = response.choices[0].message.content.strip()

    # Display overview
    st.markdown("**Product Overview:**")
    st.markdown(overview)

    # Show negative feedback themes & actions for this product
    st.markdown("---")
    st.subheader("🛠️ Negative Feedback Themes & Actions for this Product")
    df_as = compute_aspect_dataframe(filtered)
    neg = cluster_negative_feedback(df_as)
    summary = summarize_clusters(neg)
    actions = map_actions(summary)
    st.dataframe(actions, height=200)


def render_deep_analysis():
    df = load_data()
    deep_product_insights(df)
