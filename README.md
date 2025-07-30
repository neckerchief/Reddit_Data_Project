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

## 📁 Project Structure
```
Reddit_Data_Project/
├── data/                           # Only anonymized (safe for Git) raw and processed Reddit data
│   ├── raw/                        # As downloaded
│   └── processed/                  # Cleaned datasets
├── mappings/                       # NOT in Git (sensitive)
│   └── user_mapping.json           # Real → Anonymous mapping
├── notebooks/                      # Jupyter notebooks (EDA, modeling, etc.)
│   └── 01_initial_exploration.ipynb
|   └── 02_initial_exploration.ipynb
├── scripts/                        # Scripts for scraping, cleaning, feature engineering etc.
│   └── feature_engineering.py
│   └── preprocess_all.py
│   └── reddit_scraper.py
│   └── text_preprocessing.py
├── reports/                        # Reports and charts
│   └── figures/                    # Visualizations
├── README.md                       # This file
├── requirements.txt                # Dependencies
└── .gitignore
```


---

## 🛠️ Features

- ✅ Reddit data collection via API (PRAW)
- ✅ Text preprocessing and cleaning
- 🔄 Behavioral metric extraction
- 🔜 Sentiment analysis and topic modeling
- 🔜 Dashboards or reports summarizing patterns

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

## 🧠 Data Sources

 - Collected via Reddit API using praw
 - Subreddits: e.g. r/depression, r/mentalhealth, r/psychology

## 📌 Notes

The data is anonymized and used strictly for educational purposes.

This is a solo learning project, not affiliated with Reddit or any research institution.

# ✨ Author
Paulina Michalak  
[GitHub](https://github.com/neckerchief)  
[LinkedIn](https://www.linkedin.com/in/paulina-michalak-a55941252/)  
[MyPortfolio](https://neckerchief.github.io/PMichalakPortfolio.github.io/)

---
