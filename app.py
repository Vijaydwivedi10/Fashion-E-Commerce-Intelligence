import os
import base64
from io import BytesIO
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from deep_analysis import render_deep_analysis
from config import PLATFORMS, CATEGORIES, PRODUCTS
from data_utils import load_data, save_data, simulate_reviews, analyze_data
from nlp_utils import analyzer, keyword_extractor, summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

import nltk

# Download punkt only if it’s not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


from analysis_utils import (
    ASPECTS,
    compute_aspect_dataframe,
    cluster_negative_feedback,
    summarize_clusters,
    map_actions,
    generate_action_plan
)

# -------------------------------
# Page Configuration & CSS
# -------------------------------
st.set_page_config(page_title="Fashion Review Intelligence", layout="wide")
st.markdown("""
    <style>
            /* Headings */
    .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3, .css-1v0mbdj h4 {
      font-size: 0.8rem !important;      /* shrink all headings */
    }
    /* Paragraph / markdown text */
    .css-1v0mbdj p {
      font-size: 0.6rem !important;    /* shrink body text */
    }
    /* DataFrame tables */
    .css-1d391kg, .css-1d391kg th, .css-1d391kg td {
      font-size: 0.5rem !important;    /* smaller table text */
    }
    /* Plotly chart titles & axis labels */
    .js-plotly-plot .plotly .gtitle, 
    .js-plotly-plot .xtitle, 
    .js-plotly-plot .ytitle, 
    .js-plotly-plot .xtick, 
    .js-plotly-plot .ytick {
      font-size: 8px !important;      /* shrink chart fonts */
    }
      .stButton>button {
        background-color: #28a745;
        border: 2px solid #1e7e34;
        color: white;
        padding: 0.6em 1.2em;
        font-size: 0.9rem;
        border-radius: 6px;
        transition: background-color 0.2s ease, transform 0.2s ease;
      }
      .stButton>button:hover {
        background-color: #218838;
        transform: scale(1.03);
      }
      .download-btn {
        # background-color: #28a745;
        border: 2px solid green;
        color: white;
        padding: 0.6em 1.2em;
        font-size: 0.9rem;
        border-radius: 6px;
        transition: background-color 0.2s ease, transform 0.2s ease;
      }
      .download-btn:hover {
        # background-color: #db542f;
        transform: scale(1.06);
      }
      a {
        color: green !important;
      }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Session State Initialization
# -------------------------------
for key in ('sel_p','sel_prod','sel_cat','generated'):
    if key not in st.session_state:
        st.session_state[key] = False if key=='generated' else None

# -------------------------------
# Caching Expensive Computations
# -------------------------------
@st.cache_data(show_spinner=False)
def get_clusters_and_actions(data_records):
    df = pd.DataFrame.from_records(data_records)
    df_as = compute_aspect_dataframe(df)
    neg = cluster_negative_feedback(df_as)
    summary = summarize_clusters(neg)
    actions = map_actions(summary)
    plan = generate_action_plan(df)
    return actions, plan

# -------------------------------
# App Title & Login
# -------------------------------
st.title("Fashion E-Commerce Review Intelligence")
username = st.sidebar.text_input("Username")
if not username:
    st.sidebar.info("Enter username to continue.")
    st.stop()

# -------------------------------
# Load or Simulate Data
# -------------------------------
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if st.session_state.df.empty:
    if st.sidebar.button("Simulate Reviews"):
        st.session_state.df = simulate_reviews(3000)
        save_data(st.session_state.df)
        st.sidebar.success("Simulated 3000 reviews.")
    else:
        st.sidebar.warning("No data available.")
        st.stop()

df = st.session_state.df

# -------------------------------
# Submit New Review
# -------------------------------
with st.sidebar.form("new_review_form", clear_on_submit=True):
    pfrm = st.selectbox("Platform", [None] + PLATFORMS)
    prod = st.selectbox("Product", [None] + PRODUCTS)
    cat = st.selectbox("Category", [None] + CATEGORIES)
    rating = st.slider("Rating", 1, 5, 3)
    review_text = st.text_area("Review Text", height=100)
    submitted = st.form_submit_button("Submit")
if submitted:
    if None in (pfrm, prod, cat) or not review_text:
        st.sidebar.error("All fields are required.")
    else:
        dt = datetime.now()
        comp = analyzer.polarity_scores(review_text)['compound']
        sentiment = ('Positive' if comp>=0.05 else 'Negative' if comp<=-0.05 else 'Neutral')
        summary = ' '.join(str(s) for s in summarizer(
            PlaintextParser.from_string(review_text, Tokenizer("en")).document,1))
        keywords = ', '.join([kw[0] for kw in keyword_extractor.extract_keywords(review_text)])
        new_row = {"Timestamp":dt, "Platform":pfrm, "Product":prod, "User":username,
                   "Category":cat, "Rating":rating, "Review":review_text,
                   "Sentiment":sentiment, "SentimentScore":round(comp,3),
                   "Summary":summary, "Keywords":keywords}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df = df
        save_data(df)
        st.sidebar.success("Review submitted!")

# -------------------------------
# Filters & Reset Logic
# -------------------------------
st.subheader("Filters")
col1, col2, col3 = st.columns(3)
sel_p = col1.selectbox("Platform", ["All"]+PLATFORMS)
sel_prod = col2.selectbox("Product", ["All"]+PRODUCTS)
sel_cat = col3.selectbox("Category", ["All"]+CATEGORIES)

# Reset generated on filter change
for key, sel in [('sel_p',sel_p),('sel_prod',sel_prod),('sel_cat',sel_cat)]:
    if st.session_state[key] != sel:
        st.session_state[key] = sel
        st.session_state.generated = False

# Apply filters
def apply_filter(df_arg, col, sel): return df_arg if sel=="All" else df_arg[df_arg[col]==sel]

df_filt = apply_filter(apply_filter(apply_filter(df,'Platform',sel_p),'Product',sel_prod),'Category',sel_cat)

# -------------------------------
# Live Metrics
# -------------------------------
st.markdown("---")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Total Reviews", len(df_filt))
avg_sent = analyze_data(df_filt)['sentiment_over_time']['SentimentScore'].mean()
mc2.metric("Avg Sentiment", round(avg_sent,3))
mc3.metric("Negative Reviews", len(df_filt[df_filt['Sentiment']=='Negative']))
mc4.metric("Avg Rating", round(df_filt['Rating'].mean(),2))

# -------------------------------
# Charts & Tables
# -------------------------------
insights = analyze_data(df_filt)

with st.container():
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(px.pie(insights['sentiment_dist'],names='Sentiment',values='Count',hole=0.4),use_container_width=True)
    with c2:
        st.subheader("Top Complaint Keywords")
        st.plotly_chart(px.bar(insights['top_complaints'],x='Keyword',y='Frequency'),use_container_width=True)

st.markdown("---")
with st.container():
    d1,d2 = st.columns(2)
    with d1:
        st.subheader("Sentiment Over Time")
        st.plotly_chart(px.line(insights['sentiment_over_time'],x='Timestamp',y='SentimentScore'),use_container_width=True)
    with d2:
        st.subheader("Platform Breakdown")
        st.plotly_chart(px.bar(insights['by_platform'],x='Platform',y='Count',color='Sentiment',barmode='group'),use_container_width=True)

st.subheader("Category Performance")
st.dataframe(insights['category_performance'],height=200)
st.subheader("Recent Reviews")
st.dataframe(df_filt.sort_values('Timestamp',ascending=False).head(10),height=200)

# -------------------------------
# Aspect-Based Radar
# -------------------------------
st.subheader("Aspect-Based Sentiment")
df_as = compute_aspect_dataframe(df_filt)
avg = df_as[ASPECTS].mean()
labels = list(avg.index)+[avg.index[0]]; vals=list(avg.values)+[avg.values[0]]
fig = go.Figure(go.Scatterpolar(r=vals,theta=labels,fill='toself'))
fig.update_layout(polar=dict(radialaxis=dict(range=[-0.2,0.2])))
st.plotly_chart(fig,use_container_width=True)

# -------------------------------
# Generate Themes & Actions + Download
# -------------------------------

if st.button("Generate Negative Feedback Themes & Action Plans",key='gen',help='Cluster & get actions',use_container_width=True):
    st.session_state.generated = True

if st.session_state.generated:
    # cached computation
    records = df_filt.to_dict('records')
    actions_df, plan_df = get_clusters_and_actions(records)

    st.subheader("Negative Feedback Themes & Actions")
    st.dataframe(actions_df, height=200)

    # download Excel
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        plan_df.to_excel(writer, index=False, sheet_name='ActionPlan')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}' "
        f"download='action_plan.xlsx' class='download-btn'>DOWNLOAD ACTION PLAN (EXCEL FILE)</a>",
        unsafe_allow_html=True
    )

st.markdown("---")

# After your download button block:
render_deep_analysis()

# Footer
st.markdown(f"<div style='text-align:right;color:#888;'>User: {username}</div>", unsafe_allow_html=True)



