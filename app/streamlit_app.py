"""
Step 1 — Content-based movie recommender (TF-IDF + cosine similarity).
Robust `recommend_from_likes_safe()` and defensive rendering to avoid front-end crashes.
Place MovieLens small at: data/ml-latest-small/movies.csv
Run: streamlit run app/streamlit_app.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Locate data --------------------
CANDIDATES = [
    Path(__file__).resolve().parents[1] / "data" / "ml-latest-small" / "movies.csv",
    Path("data") / "ml-latest-small" / "movies.csv",
    Path("movies.csv"),
]

def find_movies_csv():
    for p in CANDIDATES:
        if p.exists():
            return p
    return None

DATA_PATH = find_movies_csv()

st.set_page_config(page_title="Movie Recommender (TF-IDF)", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender — Step 1 (Content-based)")

if DATA_PATH is None:
    st.error(
        "Could not find movies.csv. Put MovieLens ml-latest-small at:\n"
        "data/ml-latest-small/movies.csv\n\n"
        "Or upload a small movies.csv with columns: movieId,title,genres"
    )
    st.stop()

# -------------------- Load & clean data --------------------
@st.cache_data(show_spinner=False)
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"movieId", "title", "genres"}.issubset(df.columns):
        raise ValueError("movies.csv must contain columns: movieId,title,genres")
    df["genres"] = df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    df["clean_title"] = df["title"].astype(str).str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    df["doc"] = (df["clean_title"].fillna("") + " " + df["genres"]).str.lower()
    return df[["movieId", "title", "genres", "clean_title", "doc"]].reset_index(drop=True)

movies = load_and_clean(DATA_PATH)

# -------------------- Build TF-IDF --------------------
@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(docs: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X

vectorizer, X = build_vectorizer_and_matrix(movies["doc"])

# -------------------- Robust recommender for multiple likes --------------------
def recommend_from_likes_safe(movies: pd.DataFrame, X, selected_titles: list, top_k: int = 10) -> pd.DataFrame:
    """
    Robust recommender for a list of exact movie titles.
    Returns DataFrame with columns ['title','genres'] (may be empty).
    """
    # Input validation
    if selected_titles is None:
        selected_titles = []
    if not isinstance(selected_titles, (list, tuple)):
        raise TypeError("selected_titles must be a list or tuple of strings.")
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 10

    if len(selected_titles) == 0:
        return pd.DataFrame(columns=["title", "genres"])

    # map exact titles -> indices
    mask = movies["title"].isin(selected_titles)
    idxs = movies.index[mask].tolist()
    if len(idxs) == 0:
        # no exact matches
        return pd.DataFrame(columns=["title", "genres"])

    # compute profile vector (mean of item vectors)
    profile = X[idxs].mean(axis=0)

    # convert profile to a supported type for cosine_similarity
    if sparse.issparse(profile):
        query = profile  # sparse 1 x F
    else:
        query = np.asarray(profile)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        else:
            query = np.atleast_2d(query).reshape(1, -1)

    # compute similarities
    sims = cosine_similarity(query, X).ravel()  # length = n_items

    # ranking and filtering
    order = np.argsort(-sims)
    liked_set = set(idxs)
    rec_indices = [i for i in order if i not in liked_set][:top_k]

    recs = movies.iloc[rec_indices][["title", "genres"]].copy().reset_index(drop=True)

    # final safety fixes: ensure strings, unique titles, trim to top_k
    recs["title"] = recs["title"].astype(str)
    if "genres" in recs.columns:
        recs["genres"] = recs["genres"].astype(str)
    recs = recs.drop_duplicates(subset=["title"]).head(top_k).reset_index(drop=True)
    return recs

# -------------------- Single-title recommender (seed-based) --------------------
def recommend_from_title(query: str, top_k: int = 10) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame(columns=["title", "genres"])
    q = query.strip().lower()
    q_vec = vectorizer.transform([q])
    sims_to_query = cosine_similarity(q_vec, X).ravel()
    seed_idx = int(np.argmax(sims_to_query))
    seed_vec = X[seed_idx]
    sims = cosine_similarity(seed_vec, X).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != seed_idx][:top_k]
    recs = movies.iloc[order][["title", "genres"]].reset_index(drop=True)
    return recs

# -------------------- Streamlit UI --------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    mode = st.radio("Mode:", ["Single movie", "Multiple likes"])

st.write("### How to use")
st.write("- **Single movie**: type part of a title (e.g., `Matrix`) → click Recommend.")
st.write("- **Multiple likes**: pick exact titles from the list to build a tiny profile.")

if mode == "Single movie":
    title_q = st.text_input("Enter a movie you like (free-text)")
    if st.button("Recommend") and title_q:
        recs = recommend_from_title(title_q, top_k=k)
        if recs.empty:
            st.warning("No matches found. Try a different keyword or use Multiple likes.")
        else:
            st.subheader("Recommendations")
            # Defensive rendering: ensure DataFrame and simple types
            try:
                st.dataframe(recs)
            except Exception as e:
                st.error("Rendering error: " + str(e))
                st.write("Debug info:", type(recs), recs.head().to_dict())
else:
    picks = st.multiselect("Select your liked movies (exact titles)", movies["title"].tolist(), max_selections=10)
    if st.button("Recommend") and picks:
        # wrap call in try/except to surface safe errors
        try:
            recs = recommend_from_likes_safe(movies, X, picks, top_k=k)
        except Exception as e:
            st.error("Recommendation error: " + str(e))
            recs = pd.DataFrame(columns=["title", "genres"])

        if recs.empty:
            st.warning("No recommendations (maybe your picked titles were not matched).")
        else:
            st.subheader("Recommendations")
            try:
                st.dataframe(recs)
            except Exception as e:
                st.error("Rendering error: " + str(e))
                st.write("Debug info:", type(recs), recs.head().to_dict())

st.markdown("---")
st.caption("TF-IDF content-based recommender — Step 1 complete.")
