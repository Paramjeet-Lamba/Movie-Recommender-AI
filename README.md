# 🎬 Movie Recommender AI

**Pick a movie you liked. Get matches that actually fit.**

Movie Recommender AI is a content-based recommendation engine built with **Streamlit and scikit-learn** that suggests similar movies using **TF-IDF + Cosine Similarity** over movie descriptions, genre, industry, and cast. Project 4 from *AI Playground: 4 Real-World AI Projects*, it spans a multi-industry catalog (Hollywood, Bollywood, Tollywood, Lollywood) with live filtering by industry, year, genre, and actor.

### 🚀 Live Demo

**Try Movie Recommender AI:** https://movie-recommender-ai-ml.streamlit.app

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Powered%20by-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Method](https://img.shields.io/badge/Method-TF--IDF%20%2B%20Cosine%20Similarity-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Overview

Most recommender demos hard-code a similarity table and call it a day. **Movie Recommender AI** builds the similarity space live from text — no ratings, no user history, just descriptions and metadata — and layers real filtering controls on top so you can narrow the catalog before asking for matches.

* ✅ TF-IDF vectorization of description + genre + industry + cast
* 🎯 Cosine similarity scoring with a percentage match badge
* 🎛️ Live filters: Industry, Year range, Genre, Hero/Heroine
* 📚 Full catalog browser with dataset stats
* ⚡ Instant quick-example recommendations
* 🧠 Built-in "How it works" breakdown of the method

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎯 **Get Recommendations** | Pick any movie from the filtered list and get the top-N most similar titles. |
| 🎛️ **Multi-Filter Sidebar** | Narrow by Industry, Year range, Genre, and Hero/Heroine before picking a movie. |
| 📊 **Match Score Badges** | Every recommendation shows a cosine-similarity percentage match. |
| 📚 **Movie Catalog Tab** | Browse the full dataset with total movie count, filtered count, and TF-IDF vocabulary size. |
| ⚡ **Quick Examples** | Instant example recommendations for popular titles, always run against the full catalog. |
| 🌍 **Multi-Industry Dataset** | Covers Hollywood, Bollywood, Tollywood, and Lollywood titles side by side. |
| 🧠 **How It Works Tab** | Plain-English explanation of TF-IDF, cosine similarity, and filtering logic. |

---

## 🧠 How Movie Recommender AI Works

```text
                    ┌─────────────────────┐
                    │ Movie Metadata        │
                    │ Description + Genre +  │
                    │ Industry + Cast        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TF-IDF Vectorization  │
                    │ (scikit-learn)         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cosine Similarity     │
                    │ Matrix (all pairs)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Apply Sidebar Filters │
                    │ Industry/Year/Genre/  │
                    │ Actor                 │
                    └──────────┬──────────┘
                               │
                               ▼
                    Top-N Ranked Matches
```

TF-IDF converts each movie's enriched text into a vector of weighted word scores — a word common in one movie's text but rare across the rest of the catalog gets a higher weight. Cosine similarity then compares every movie's vector against every other movie's vector, producing a score between 0 (unrelated) and 1 (identical) based on the angle between them. The similarity matrix is computed once across the full catalog; the sidebar filters only control which movies are eligible to be picked from and recommended.

---

## 🛠️ Tech Stack

### Frontend & Application
* **Python 3.11**
* **Streamlit** — Web application framework

### Machine Learning
* **scikit-learn** — `TfidfVectorizer` and `cosine_similarity`

### Data Processing
* **Pandas** — Dataset handling, filtering, and catalog display

### Deployment
* **Streamlit Cloud** — Application hosting

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Paramjeet-Lamba/Movie-Recommender-AI.git
cd Movie-Recommender-AI
```

### 2. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Run Movie Recommender AI

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
Movie-Recommender-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── config.toml
```

---

## 🧪 Usage

### Step 1 — Filter the Catalog (optional)

Use the sidebar to narrow by **Industry**, **Year range**, **Genre**, and **Hero/Heroine**.

### Step 2 — Pick a Movie

Select any title from the filtered list on the **🎯 Get Recommendations** tab.

### Step 3 — Get Matches

Click **🎬 Recommend Similar Movies** to see the top matches, each with industry, year, genre, cast, and a similarity match badge.

### Step 4 — Explore the Catalog

Switch to **📚 Movie Catalog** to browse the full dataset and see how many movies match your current filters.

### Step 5 — Learn the Method

The **🧠 How It Works** tab breaks down TF-IDF, cosine similarity, and how filtering interacts with the similarity matrix.

---

## ⚠️ Important Disclaimer

Movie Recommender AI is a **content-based** recommender — it only looks at text descriptions and metadata (genre, industry, cast), not user behavior or ratings. Real-world recommendation systems usually combine this approach with **collaborative filtering** (what similar users liked) for stronger results. Match percentages reflect text similarity, not a guarantee you'll enjoy the recommended title.

---

## 🔮 Future Improvements

Potential improvements for future versions include:

* ⭐ Collaborative filtering layer using user ratings
* 🌐 Larger, real-world movie dataset via an external API
* 🎨 Poster images and richer movie detail cards
* 📈 Personalized recommendation history per session
* 🔗 Hybrid content + collaborative scoring
* 📱 Mobile-friendly layout refinements

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you have an idea that can improve Movie Recommender AI, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">

### 🎬 Movie Recommender AI

**Descriptions → TF-IDF → Cosine Similarity → Top Matches**

Built with ❤️ using **Python, Streamlit & scikit-learn**

</p>
