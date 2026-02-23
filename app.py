import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬")

st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
    }
    .stButton>button {
        background-color: #22C55E;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .movie-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Content-Based Movie Recommendation System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies["genres"] = movies["genres"].fillna("")
    movies["title"] = movies["title"].str.strip()
    return movies, ratings

movies, ratings = load_data()

# -----------------------------
# Reduce Dataset and Deduplicate
# -----------------------------
movies = movies.drop_duplicates(subset="title").reset_index(drop=True)
movies = movies.head(1500)  # Reduce memory for free-tier hosting
indices = pd.Series(movies.index, index=movies["title"])

# -----------------------------
# Build Cached TF-IDF
# -----------------------------
@st.cache_resource
def build_tfidf(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["genres"])
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_tfidf(movies)

# -----------------------------
# Content-Based Recommendation
# -----------------------------
def get_content_candidates(title, top_k=20):
    if title not in indices:
        return []

    idx = indices[title]

    # Compute similarity only for selected movie
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:top_k+1]

    return [(i, sim_scores[i]) for i in sim_indices]

# -----------------------------
# Hybrid Recommendation
# -----------------------------
def hybrid_recommendation(title, user_id, alpha=0.5, n=5):
    # Cold start handling
    if user_id not in ratings["userId"].unique():
        st.warning("New user detected. Showing content-based results only.")
        content_only = get_content_candidates(title, n)
        return [movies.iloc[i[0]]["title"] for i in content_only]

    content_candidates = get_content_candidates(title, 20)

    if not content_candidates:
        return ["Movie not found"]

    # Movies already rated by user
    rated_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    hybrid_scores = []

    for idx, sim_score in content_candidates:
        movie_id = movies.iloc[idx]["movieId"]

        # Skip already watched
        if movie_id in rated_movies:
            continue

        final_score = sim_score
        hybrid_scores.append((idx, final_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    top_movies = hybrid_scores[:n]

    return [movies.iloc[i[0]]["title"] for i in top_movies]

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Recommendation Settings")

movie_input = st.sidebar.selectbox(
    "🎬 Select a Movie",
    sorted(movies["title"].unique())
)

user_id = st.sidebar.number_input(
    "👤 Enter User ID",
    min_value=1,
    step=1
)

alpha = st.sidebar.slider(
    "⚖️ Content vs Collaborative",
    0.0, 1.0, 0.5
)

top_n = st.sidebar.slider(
    "🔢 Number of Recommendations",
    1, 10, 5
)

recommend_button = st.sidebar.button("🚀 Recommend Movies")

# -----------------------------
# Recommendation Display
# -----------------------------
if recommend_button:
    st.markdown("## 🎯 Recommended For You")
    recommendations = hybrid_recommendation(
        movie_input,
        user_id,
        alpha,
        top_n
    )

    for rec in recommendations:
        st.markdown(f"""
            <div class="movie-card">
                <h4>🎥 {rec}</h4>
            </div>
        """, unsafe_allow_html=True)