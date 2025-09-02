# 📊 Brand Reputation Monitor

This project monitors **real-time news articles** about entertainment brands (e.g., Netflix, Disney, Prime Video),  
analyzes **sentiment & emotion**, and visualizes the results in an **interactive Streamlit dashboard**.  

It helps track **brand reputation** by detecting **positive/negative trends**,  
highlighting anomalies, and visualizing emotions with charts & word clouds.

---

## 🚀 Features

✅ Collects real-time news data (via **NewsAPI**)  
✅ Performs **sentiment & emotion analysis**  
✅ Generates **aspect-based sentiment insights**  
✅ Visualizes results in a **Streamlit dashboard**  
✅ Supports **word clouds, anomaly detection, and radar charts**  

---

## 📂 Project Structure


brand-reputation-monitor/
│── src/
│   ├── fetch_realtime.py       # Fetches news data
│   ├── sentiment_realtime.py   # Runs sentiment + emotion analysis
│   ├── dashboard.py            # Streamlit dashboard
│
│── data/
│   ├── raw/                    # Raw API responses
│   ├── processed/              # Processed CSVs with sentiment/emotion
│
│── requirements.txt            # Python dependencies
│── .env.example                # Example env file (no secrets)
│── .gitignore                  # Ignore unnecessary files
│── README.md                   # Project documentation



## 🛠️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/brand-reputation-monitor.git
   cd brand-reputation-monitor

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   Set up environment variablesCreate a .env file in the root directory:

3. **env**

NEWSAPI_KEY=your_api_key_here

4. **Run sentiment analysis pipeline**

   ```bash
   python src/sentiment_realtime.py

5. **Run Streamlit Dashboard**

   ```bash
   streamlit run src/dashboard.py


📊 Example Dashboard
📈 Sentiment Trends (positive/negative/neutral over time)
🎭 Emotion Radar Chart (joy, anger, sadness, etc.)
☁️ Word Clouds (brand-wise highlights)
⚠️ Anomaly Detection (sudden spikes in negative sentiment)

📦 Dependencies
streamlit
pandas
numpy
requests
python-dotenv
nltk
scikit-learn
matplotlib
plotly
wordcloud

⚠️ Notes
Do NOT upload your .env with the real API key to GitHub.
Only .env.example should be shared.
Processed CSV files (data/processed/) are ignored from version control.

🧑‍💻 Contributors
Magna Benita P

📜 License
This project is licensed under the MIT License – free to use and modify.
