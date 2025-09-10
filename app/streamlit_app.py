"""
Movie Recommender App — TF-IDF content-based (patched UX)
- Single-movie input now in a form so pressing Enter submits.
- Adds similarity threshold to avoid returning unrelated matches for nonsense queries.
- Shows warnings when no picks provided in Multiple-likes.
- Displays 1-based index for recommendations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- File check --------------------
REQUIRED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
CANDIDATE_DIRS = [
    Path(__file__).resolve().parents[1] / "data" / "ml-latest-small",
    Path("data") / "ml-latest-small",
    Path("ml-latest-small"),
]

def find_movielens_files():
    for d in CANDIDATE_DIRS:
        files = {f: d / f for f in REQUIRED_FILES}
        if all(path.exists() for path in files.values()):
            return files
    return None

DATA_FILES = find_movielens_files()

st.set_page_config(page_title="Movie Recommender (TF-IDF)", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender — TF-IDF (Step 1)")

if DATA_FILES is None:
    st.error(
        "❌ Could not find full MovieLens dataset.\n\n"
        "Please put the following files under `data/ml-latest-small/`:\n"
        "- movies.csv\n- ratings.csv\n- tags.csv\n- links.csv"
    )
    st.stop()

# -------------------- Load & clean movies --------------------
@st.cache_data(show_spinner=False)
def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genres"] = df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    df["clean_title"] = df["title"].astype(str).str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    df["doc"] = (df["clean_title"].fillna("") + " " + df["genres"]).str.lower()
    return df[["movieId", "title", "genres", "clean_title", "doc"]].reset_index(drop=True)

movies = load_movies(DATA_FILES["movies.csv"])

# -------------------- Build TF-IDF --------------------
@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(docs: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X

vectorizer, X = build_vectorizer_and_matrix(movies["doc"])

# -------------------- Recommender functions --------------------
def recommend_from_likes_safe(movies: pd.DataFrame, X, selected_titles: list, top_k: int = 10) -> pd.DataFrame:
    if not selected_titles:
        return pd.DataFrame(columns=["title", "genres"])
    mask = movies["title"].isin(selected_titles)
    idxs = movies.index[mask].tolist()
    if not idxs:
        return pd.DataFrame(columns=["title", "genres"])
    profile = X[idxs].mean(axis=0)
    if sparse.issparse(profile):
        query = profile
    else:
        query = np.asarray(profile)
        query = np.atleast_2d(query)
    sims = cosine_similarity(query, X).ravel()
    order = np.argsort(-sims)
    rec_indices = [i for i in order if i not in idxs][:top_k]
    recs = movies.iloc[rec_indices][["title", "genres"]].copy().reset_index(drop=True)
    return recs

def recommend_from_title_with_threshold(query: str, top_k: int = 10, min_sim: float = 0.12):
    if not query or not query.strip():
        return None, None
    q = query.strip().lower()
    q_vec = vectorizer.transform([q])
    sims_to_query = cosine_similarity(q_vec, X).ravel()
    best_sim = float(np.max(sims_to_query))
    seed_idx = int(np.argmax(sims_to_query))
    if best_sim < min_sim:
        return None, best_sim
    seed_vec = X[seed_idx]
    sims = cosine_similarity(seed_vec, X).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != seed_idx][:top_k]
    recs = movies.iloc[order][["title", "genres"]].reset_index(drop=True)
    return recs, best_sim

# -------------------- UI: Sidebar settings --------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    min_sim = st.slider("Similarity threshold (single-text)", min_value=0.00, max_value=0.40, value=0.12, step=0.01,
                        help="If the input text has low similarity to all movies, the app will warn instead of returning unrelated results.")
    mode = st.radio("Mode:", ["Single movie", "Multiple likes"])
    st.caption("Tip: press Enter inside the text field to submit when using Single movie (form submit).")

st.write("### How to use")
st.write("- **Single movie**: type part of a title (e.g., `Matrix`) and press Enter or click Recommend.")
st.write("- **Multiple likes**: pick exact titles from the list to build a tiny profile and click Recommend.")

# -------------------- Single movie (form so Enter submits) --------------------
if mode == "Single movie":
    with st.form(key="single_movie_form", clear_on_submit=False):
        title_q = st.text_input("Enter a movie you like (free-text)", value="", key="single_input")
        submitted = st.form_submit_button("Recommend")
        if submitted:
            recs, best_sim = recommend_from_title_with_threshold(title_q, top_k=k, min_sim=min_sim)
            if recs is None:
                if best_sim is None:
                    st.warning("Please type a movie title (or try Multiple likes).")
                else:
                    st.warning(f"No strong match found (best similarity {best_sim:.3f} < threshold {min_sim:.2f}). Try a different query or use Multiple likes.")
            elif recs.empty:
                st.info("No recommendations found.")
            else:
                st.subheader("Recommendations")
                recs_display = recs.copy()
                recs_display.index = range(1, len(recs_display) + 1)
                st.dataframe(recs_display)

# -------------------- Multiple likes --------------------
else:
    picks = st.multiselect("Select your liked movies (exact titles)", movies["title"].tolist(), max_selections=10)
    if st.button("Recommend"):
        if not picks:
            st.warning("Please select one or more movies (use Multiple likes to pick exact titles).")
        else:
            try:
                recs = recommend_from_likes_safe(movies, X, picks, top_k=k)
            except Exception as e:
                st.error("Recommendation error: " + str(e))
                recs = pd.DataFrame(columns=["title", "genres"])
            if recs.empty:
                st.warning("No recommendations (maybe your picks were not matched).")
            else:
                st.subheader("Recommendations")
                recs_display = recs.copy()
                recs_display.index = range(1, len(recs_display) + 1)
                st.dataframe(recs_display)

st.markdown("---")
st.caption("TF-IDF content-based recommender — Step 1 (patched).")
