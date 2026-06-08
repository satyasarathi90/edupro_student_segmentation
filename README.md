# EduPro • Student Segmentation & Personalized Course Recommendation System

Streamlit app that:
- Builds learner profiles from `users.csv`, `courses.csv`, `transactions.csv`
- Segments learners using K-Means clustering (feature-engineered profiles)
- Recommends top-k courses per learner using cluster-aware relevance scoring

## Folder layout
- `app.py` — Streamlit UI
- `data/` — CSVs: `users.csv`, `courses.csv`, `transactions.csv`
- `src/` — feature engineering, segmentation, recommendations, evaluation
- `docs/` — research paper + executive summary
- `scripts/` — Windows run script

## Setup (Windows)
```bat
cd C:/Users/badal/Desktop/edupro_student_segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bat
streamlit run app.py --server.port 8503 --server.headless=true --browser.serverAddress=127.0.0.1
```
Or:
```bat
scripts\run_streamlit.bat
```

Open: http://localhost:8503

## Expected CSV columns
- `users.csv`: `UserID, Age, Gender`
- `courses.csv`: `CourseID, CourseCategory, CourseType, CourseLevel, CourseRating`
- `transactions.csv`: `UserID, CourseID, TransactionDate, Amount`

Upload alternatives in the Streamlit sidebar if needed.

