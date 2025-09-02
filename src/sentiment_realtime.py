#src/sentiment_realtime.py
import os
import json
import time
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from dotenv import load_dotenv
from transformers import pipeline

# Load environment
load_dotenv()

RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load Hugging Face emotion model once
print("🔄 Loading emotion analysis model...")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
print("✅ Emotion model ready!")

def analyze_sentiment(text):
    """Return polarity score and label (pos/neg/neu)."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return polarity, label

def analyze_emotion(text):
    """Return dominant emotion label and confidence score."""
    try:
        results = emotion_pipeline(text[:512])  # limit input length
        if isinstance(results, list) and len(results) > 0:
            res = results[0][0]
            return res["label"], res["score"]
    except Exception as e:
        print("⚠️ Emotion analysis failed:", e)
    return "unknown", 0.0

def process_articles():
    """Continuously process new articles in real time with advanced analysis."""
    print("🚀 Real-time sentiment + emotion analysis started...")
    seen = set()

    while True:
        for filename in os.listdir(RAW_DATA_DIR):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(RAW_DATA_DIR, filename)
            output_file = os.path.join(PROCESSED_DIR, filename.replace(".jsonl", "_sentiment.csv"))

            rows = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    article = json.loads(line)
                    url = article.get("url")

                    if url in seen:
                        continue
                    seen.add(url)

                    text = f"{article.get('title', '')} {article.get('description', '')}"

                    # Basic sentiment
                    polarity, sentiment_label = analyze_sentiment(text)

                    # Advanced emotion analysis
                    emotion_label, emotion_score = analyze_emotion(text)

                    rows.append({
                        "brand": "_".join(filename.split("_")[:-1]),
                        "title": article.get("title"),
                        "url": url,
                        "publishedAt": article.get("publishedAt"),
                        "sentiment": sentiment_label,
                        "polarity": polarity,
                        "emotion": emotion_label,
                        "emotion_score": emotion_score
                    })

            if rows:
                df = pd.DataFrame(rows)
                if os.path.exists(output_file):
                    df.to_csv(output_file, mode="a", header=False, index=False)
                else:
                    df.to_csv(output_file, index=False)

                print(f"✅ Processed {len(rows)} new articles from {filename}")

        print("⏳ Waiting 2 minutes before next check...")
        time.sleep(120)

if __name__ == "__main__":
    process_articles()
