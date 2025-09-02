# src/dashboard.py
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"

def load_latest_data():
    """Load all processed CSVs and aggregate sentiment/emotion stats."""
    all_dfs = []
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_sentiment.csv")]

    for f in files:
        try:
            df = pd.read_csv(os.path.join(PROCESSED_DIR, f), on_bad_lines='skip')
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to read {f}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure datetime
    if "publishedAt" in combined.columns:
        combined["publishedAt"] = pd.to_datetime(combined["publishedAt"], errors='coerce')

    return combined

def plot_heatmap(df, selected_brand):
    df['hour'] = df['publishedAt'].dt.hour
    heatmap_data = df.groupby(['brand', 'hour'])['overall_sent_score'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='brand', columns='hour', values='overall_sent_score')

    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Brand", color="Avg Sentiment"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_radar_chart(df):
    # Select emotion columns that exist
    emotion_cols = [col for col in ['overall_sent_score', 'joy', 'anger', 'sadness', 'fear'] if col in df.columns]
    if not emotion_cols:
        st.warning("No emotion columns found for radar chart.")
        return

    # Compute average per brand
    radar_data = df.groupby('brand')[emotion_cols].mean().reset_index()

    # Melt dataframe from wide to long format
    radar_long = radar_data.melt(id_vars='brand', value_vars=emotion_cols,
                                 var_name='emotion', value_name='score')

    # Plot
    fig = px.line_polar(radar_long, r='score', theta='emotion', color='brand', line_close=True,
                        markers=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud(df):
    if 'title' not in df.columns or 'overall_sent_score' not in df.columns:
        st.warning("No 'title' or sentiment score column for word cloud.")
        return

    
    pos_text = " ".join(df[df['overall_sentiment'] == "positive"]['title'].tolist())
    neg_text = " ".join(df[df['overall_sentiment'] == "negative"]['title'].tolist())
    
    if pos_text:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
        st.image(wordcloud_pos.to_array(), caption="Positive Sentiment Words", use_container_width=True)
    else:
        st.warning("No positive words to plot!")

    if neg_text:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
        st.image(wordcloud_neg.to_array(), caption="Negative Sentiment Words", use_container_width=True)
    else:
        st.warning("No negative words to plot!")


def plot_anomaly_detection(df):
    # Filter negative sentiment
    df_neg = df[df['overall_sentiment'] == "negative"].copy()


    # If no negative data, simulate some for demonstration
    if df_neg.empty:
        st.info("No negative sentiment data — adding demo negative data for visualization.")
        import numpy as np
        hours = np.arange(0, 24)
        scores = np.random.randint(-5, -1, size=24)  # simulated negative scores
        df_neg = pd.DataFrame({'hour': hours, 'overall_sent_score': scores})

    else:
        df_neg['hour'] = df_neg['publishedAt'].dt.hour

    # Aggregate counts per hour
    df_neg_hourly = df_neg.groupby('hour')['overall_sent_score'].count().reset_index()

    # Rolling mean & std for anomaly detection
    rolling_mean = df_neg_hourly['overall_sent_score'].rolling(window=3, min_periods=1).mean()
    rolling_std = df_neg_hourly['overall_sent_score'].rolling(window=3, min_periods=1).std()
    df_neg_hourly['anomaly'] = (df_neg_hourly['overall_sent_score'] - rolling_mean) > (2 * rolling_std.fillna(0))

    # Plot
    fig = px.line(df_neg_hourly, x='hour', y='overall_sent_score', markers=True, title="Negative Sentiment Spike Detection (Demo)")
    fig.add_scatter(
        x=df_neg_hourly[df_neg_hourly['anomaly']]['hour'],
        y=df_neg_hourly[df_neg_hourly['anomaly']]['overall_sent_score'],
        mode='markers', marker=dict(color='red', size=10), name='Anomaly'
    )
    st.plotly_chart(fig)



def plot_multibrand_sentiment(df, sentiment_filter="all"):
    """
    Plots average sentiment score over time for multiple brands,
    optionally filtered by sentiment type.
    Also shows the news articles that contributed to sentiment drops.
    
    Parameters:
    - df: DataFrame containing ['brand', 'publishedAt', 'overall_sent_score', 'title', 'url']
    - sentiment_filter: 'all', 'positive', 'negative'
    """
    df = df.copy()

    # Filter by sentiment
    if sentiment_filter == "positive":
        df = df[df['overall_sent_score'] > 0]
    elif sentiment_filter == "negative":
        df = df[df['overall_sent_score'] < 0]

    if df.empty:
        st.warning(f"No {sentiment_filter} sentiment data available.")
        return

    # Ensure 'publishedAt' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['publishedAt']):
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])

    # Extract hour and calculate average sentiment per brand
    df['hour'] = df['publishedAt'].dt.hour
    trend = df.groupby(['brand', 'hour'])['overall_sent_score'].mean().reset_index()

    # Plot line chart
    fig = px.line(
        trend,
        x='hour',
        y='overall_sent_score',
        color='brand',
        markers=True,
        title=f"Brand Sentiment Trend by Hour ({sentiment_filter.title()})",
        labels={'overall_sent_score': 'Average Sentiment', 'hour': 'Hour of Day'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Identify the hour(s) with lowest sentiment per brand
    st.subheader("News Contributing to Sentiment Drops")
    for brand in df['brand'].unique():
        brand_df = df[df['brand'] == brand]
        min_hour = brand_df.groupby('hour')['overall_sent_score'].mean().idxmin()
        drop_articles = brand_df[brand_df['hour'] == min_hour].sort_values('overall_sent_score')

        st.markdown(f"**{brand}** - Hour with lowest sentiment: {min_hour}:00")
        for _, row in drop_articles.iterrows():
            st.markdown(f"- [{row['title']}]({row['url']}) (Sentiment: {row['overall_sent_score']:.2f})")


def main():
    st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")
    st.title("📊 Brand Reputation Dashboard (Real-time)")

    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60 * 1000, key="dashboard_refresh")
    st.info("Data refreshes every 60 seconds. Make sure `sentiment_realtime.py` is running!")

    df = load_latest_data()
    if df.empty:
        st.warning("No data available yet. Run the real-time processing script first.")
        return

    brands = df['brand'].unique()
    selected_brand = st.selectbox("Select Brand", brands)
    brand_df = df[df['brand'] == selected_brand]

    # Basic Sentiment & Emotion Charts
    st.subheader(f"Sentiment Distribution for {selected_brand}")
    if 'overall_sentiment' in brand_df.columns:
        st.bar_chart(brand_df['overall_sentiment'].value_counts())

    st.subheader(f"Emotion Distribution for {selected_brand}")
    if 'emotion' in brand_df.columns:
        st.bar_chart(brand_df['emotion'].value_counts())

    # Tabs for new dashboard features
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heatmap", "Radar Chart", "Word Cloud", "Anomaly Detection","Multi-Brand Sentiment Trend"])

    with tab1:
        st.subheader(f"Sentiment Heatmap by Hour for {selected_brand}")
        plot_heatmap(df, selected_brand)

    with tab2:
        st.subheader("Brand Comparison Radar Chart")
        plot_radar_chart(df)

    with tab3:
        st.subheader("Trending Keywords Word Cloud")
        plot_wordcloud(brand_df)

    with tab4:
        st.subheader(f"Anomaly Detection for {selected_brand}")
        plot_anomaly_detection(brand_df)

    
    with tab5:
        st.subheader("Compare Brands Over Time")
        sentiment_option = st.radio("Filter Sentiment:", options=["all", "positive", "negative"], horizontal=True)
        plot_multibrand_sentiment(df, sentiment_filter=sentiment_option)



    # Recent Articles Table
    st.subheader(f"Recent Articles for {selected_brand}")
    display_cols = ['publishedAt', 'title', 'overall_sentiment', 'overall_sent_score', 'emotion', 'emotion_score', 'url']
    display_cols = [col for col in display_cols if col in brand_df.columns]
    st.dataframe(brand_df[display_cols].sort_values(by='publishedAt', ascending=False))

if __name__ == "__main__":
    main()
