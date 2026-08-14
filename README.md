# Movie Recommender AI — Streamlit App

Project 4 from *AI Playground: 4 Real-World AI Projects* — deployed as a
standalone, interactive web app. A content-based recommender using
TF-IDF + cosine similarity over short movie descriptions.

Developed by **Paramjeet Lamba**

## Files

- `app.py` — the full Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme + server settings (fixes the
  "browser doesn't open automatically" issue)

## Setup (VS Code / local)

```bash
# 1. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app should open automatically at `http://localhost:8501`. If it
doesn't (common on WSL/SSH/remote setups with no browser handler),
open that address manually.

## What it does

1. Pick a movie you liked from the dropdown.
2. Click **Recommend Similar Movies** — the app converts every movie's
   description into a TF-IDF vector, compares them with cosine
   similarity, and shows the closest matches with a match percentage.
3. Adjust how many recommendations to show from the sidebar.
4. Browse the full 12-movie catalog and see how the engine works in
   the other tabs.
