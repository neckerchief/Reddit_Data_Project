# REDDIT DATA PROJECT 🚀 (In Progress)

This is an ongoing personal data science project exploring Reddit content and user behavior patterns, with a focus on cognitive, psychological, and social insights.

![Status](https://img.shields.io/badge/status-in--progress-yellow)

---

## 🧠 Project Goal

This project aims to simulate a full data science workflow from raw data collection to analysis and interpretation. It focuses on Reddit posts (e.g., from r/depression, r/mentalhealth, etc.) and seeks to explore:

- Behavioral features (e.g., posting frequency, timing)
- Textual content (themes, word usage, sentiment)
- Engagement metrics (votes, comments)
- Mental health signals from language

---

## 📊 Data Sources

The project uses the following datasets:

### Historical Data
- **merged_chunks_processed.parquet** (1.72GB)
  - Historical r/depression and r/mentalhealth posts from 2019 to 2024
  
  **How to acquire data:**
  1. Download `depression_submissions.zst` and `mentalhealth_submission.zst` from [Academic Torrents](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4) and place in `data/raw/zstd/` folder
  2. Run `past_scraper.py` to parse .zst files (automatic user anonymization included) → creates `reddit_posts_historical_master.csv` in `data/raw/zst/`
  3. Run `preprocess_past.py` → processes data in chunks and saves them to `data/processed/processed_chunks/` as `processed_chunk_i.parquet` → merges chunks into `merged_chunks_processed.parquet` in `data/processed/`
### Live Data Collection
- **reddit_posts_master_processed.parquet** 
  - Collected via Reddit API using PRAW
  - File updated daily with `automated_daily_scraper.bat`
  
  **Pipeline:**
  - `reddit_scraper.py` → scrapes newest/hottest posts from a given day, anonymizes user IDs, and saves to `data/raw/reddit_posts_master.csv`
  - `preprocess_all.py` → processes text cells (`text_preprocessing.py`) in `reddit_posts_master.csv` and creates new features (`feature_engineering.py`), saving results in `data/processed/reddit_posts_master_processed.parquet`

---

## 📁 Project Structure
```
Reddit_Data_Project/
├── data/                           # Anonymized Reddit data (not included in Git due to size)
│   ├── raw/                        # Raw downloaded data
│   │   ├── zstd/                   # Historical .zst files
│   │   └── reddit_posts_master.csv
│   └── processed/                  # Cleaned and processed datasets
│       ├── processed_chunks/       # Individual processed chunks
│       ├── merged_chunks_processed.parquet
│       └── reddit_posts_master_processed.parquet
├── mappings/                       # NOT in Git (sensitive data)
│   └── user_mapping.json           # Real → Anonymous mapping
├── notebooks/                      # Jupyter notebooks (EDA, modeling, etc.)
│   └── 01_initial_exploration.ipynb
|   └── long_short_posts.ipynb
├── scripts/                        # Data processing and collection scripts
│   ├── automated_daily_scraper.bat # Daily data collection automation
│   ├── feature_engineering.py      # Feature creation and extraction
│   ├── past_scraper.py            # Historical data parsing
│   ├── preprocess_all.py          # Main preprocessing pipeline
│   ├── preprocess_past.py         # Historical data preprocessing
│   ├── reddit_scraper.py          # Live Reddit API scraping
│   └── text_preprocessing.py      # Text cleaning and NLP preprocessing
├── reports/                        # Reports and charts
│   └── figures/                    # Visualizations
├── README.md                       # This file
├── requirements.txt                # Dependencies
└── .gitignore
```


---

## 🛠️ Features & Progress

- ✅ **Data Collection**: Reddit API integration (PRAW) + historical data processing
- ✅ **Text Preprocessing**: Cleaning, tokenization, and NLP preparation
- ✅ **Automated Pipeline**: Daily data collection and processing
- ✅ **User Privacy**: Complete anonymization of user identities
- 🔄 **Feature Engineering**: Behavioral and textual metrics extraction
- 🔄 **Exploratory Analysis**: Initial data exploration and pattern discovery
- 🔜 **Sentiment Analysis**: Emotion and sentiment detection
- 🔜 **Topic Modeling**: Thematic analysis of discussions
- 🔜 **Visualization Dashboard**: Interactive reports and insights
- 🔜 **Statistical Analysis**: Correlation and behavioral pattern analysis
---

## 📦 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Reddit_Data_Project.git
   cd Reddit_Data_Project
   ``` 
2. (Optional) Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## 📌 Notes

The data is anonymized and used strictly for educational purposes.

This is a solo learning project, not affiliated with Reddit or any research institution.

# ✨ Author
Paulina Michalak  
[GitHub](https://github.com/neckerchief)  
[LinkedIn](https://www.linkedin.com/in/paulina-michalak-a55941252/)  
[MyPortfolio](https://neckerchief.github.io/PMichalakPortfolio.github.io/)

---
