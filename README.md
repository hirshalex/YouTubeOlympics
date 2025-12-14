# YouTubeOlympics

This project analyzes whether **Olympic events** and **country-level medal performance** influence the visibility of **sports-related YouTube videos**, using a large-scale dataset of Trending videos across 11 countries (2020–2024). We construct a multilingual sports classifier, label millions of videos, and estimate the causal impact of Olympic periods on the share of sports content appearing on YouTube Trending.

---

## Overview

We study two primary questions:

1. **Do the Olympic Games causally increase the prominence of sports content on YouTube Trending?**  
2. **Does a country’s Olympic medal performance further amplify local interest in sports videos?**

Our analysis combines:

- A multilingual labeling pipeline using an LLM-trained sports classifier  
- 2.9M YouTube trending video records across 11 countries  
- Medal datasets from the 2020 Summer and 2022 Winter Olympics  
- Regression and rolling-window designs to identify Olympic and medal effects  

---

## Data Sources

### YouTube Trending Videos (Kaggle)
Dataset updated daily, containing video metadata across 11 countries.  
<https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset>

### Olympic Medal Data (Kaggle)
**Tokyo 2020 Summer Games:**  
<https://www.kaggle.com/datasets/piterfm/tokyo-2020-olympics>

**Beijing 2022 Winter Games:**  
<https://www.kaggle.com/datasets/piterfm/beijing-2022-olympics>

---

## Methodology

### 1. Sports Label Construction
- LLM-generated labels for a curated multilingual sample  
- TF-IDF or embedding-based classifier used to label all 2.9M videos  
- Final label categories: *Olympic*, *Other Sport*, *Non-Sport*

### 2. Causal Analysis
- Staggered before–after regressions around Olympic periods  
- Rolling-window medal exposure tests for medal effects  
- Country fixed effects + seasonality controls

### 3. Outcomes Analyzed
- Share of sports videos appearing on YouTube Trending  
- Distinction between Olympic-specific videos and general sports spillover patterns  

---

## Key Findings (Brief)

- Olympic periods produce a **large and statistically significant increase** in sports-related trending videos.  
- The Summer Olympics generate the largest effect: **≈ +13 percentage points** at peak.  
- Winter Olympics show smaller but still meaningful increases.  
- **Medal counts do not meaningfully influence sports visibility**, indicating that global Olympic attention—not national success—drives the surge.

---

## Repository Structure
YouTubeOlympics/
│
├── data/
│ ├── raw/ # Original Kaggle CSVs
│ ├── processed/ # Parquet files after cleaning + labeling
│
├── scripts/
│ ├── merge_csvs.py
│ ├── classify_tfidf.py
│ ├── label_full.py
│
├── notebooks/
│ ├── EDA.ipynb
│ ├── Modeling.ipynb
│ ├── CausalEffects.ipynb
│
├── models/
│ └── tfidf_student.joblib
│
└── README.md

SLURM-based distributed jobs for large-scale labeling are included under scripts/.

## Authors

Alex Hirsh, Hayden King, Raybal Ahmad
