#src/fetch_realtime.py
import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load API key
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

BASE_URL = "https://newsapi.org/v2/everything"
BRANDS = ["Netflix", "Disney+", "Prime Video", "Warner Bros Discovery", "TikTok", "Spotify"]


RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Keep track of seen URLs to avoid duplicates
seen_urls = set()

def fetch_news(brand, page_size=20):
    """Fetch latest news for a brand."""
    params = {
        "q": brand,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data.get("articles", [])

def save_articles(brand, articles):
    """Append new articles to file, skip already seen ones."""
    global seen_urls
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    filename = f"{RAW_DATA_DIR}/{brand.lower().replace(' ', '_').replace('+','plus')}_{timestamp}.jsonl"

    new_count = 0
    with open(filename, "a", encoding="utf-8") as f:
        for article in articles:
            url = article.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                json.dump(article, f, ensure_ascii=False)
                f.write("\n")
                new_count += 1

    if new_count > 0:
        print(f"✅ Added {new_count} new articles for {brand}")
    else:
        print(f"ℹ️ No new articles for {brand}")

if __name__ == "__main__":
    print("🚀 Starting real-time news monitoring...")
    while True:
        for brand in BRANDS:
            articles = fetch_news(brand, page_size=50)
            save_articles(brand, articles)

        print("⏳ Waiting 5 minutes before next fetch...")
        time.sleep(300)  # 5 minutes



