"""
Streamlit app: TF-IDF + ALS + Hybrid with TMDb posters & summaries (optional).

Requirements:
- TMDb key (optional): store as Streamlit secret 'TMDB_API_KEY' OR environment variable TMDB_API_KEY.
- links.csv must be present in data/ml-latest-small/links.csv mapping movieId -> tmdbId.

If TMDb key or mapping missing, the app falls back to text-only recommendations.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import requests
from requests.exceptions import RequestException
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-latest-small"
MODELS_DIR = ROOT / "models"
LINKS_PATH = DATA_DIR / "links.csv"

# -------------------- Page config --------------------
st.set_page_config(page_title="Movie Recommender (TF-IDF + ALS)", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender — TF-IDF + ALS (Hybrid + Posters)")

# -------------------- Helper: files --------------------
def require_file(p: Path, what: str):
    if not p.exists():
        st.error(f"Missing {what}: {p}\nPlease put the required file and reload the app.")
        st.stop()

require_file(DATA_DIR / "movies.csv", "movies.csv")

# -------------------- Load movies and TF-IDF --------------------
@st.cache_data(show_spinner=False)
def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"movieId", "title", "genres"}.issubset(df.columns):
        raise ValueError("movies.csv must contain columns: movieId,title,genres")
    df = df.copy()
    df["genres"] = df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    df["clean_title"] = df["title"].astype(str).str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    df["doc"] = (df["clean_title"].fillna("") + " " + df["genres"]).str.lower()
    return df[["movieId", "title", "genres", "clean_title", "doc"]].reset_index(drop=True)

movies = load_movies(DATA_DIR / "movies.csv")

@st.cache_resource(show_spinner=False)
def build_tfidf(docs: pd.Series):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(docs)
    return vec, X

tfidf_vec, X = build_tfidf(movies["doc"])

# -------------------- Load links.csv mapping (movieId -> tmdbId) --------------------
@st.cache_data(show_spinner=False)
def load_links(path: Path):
    if not path.exists():
        return {}
    links = pd.read_csv(path)
    # tmdbId column may be object; coerce to numeric, drop na
    if "tmdbId" not in links.columns:
        return {}
    links = links[["movieId", "tmdbId"]].dropna()
    links["movieId"] = links["movieId"].astype(int)
    # tmdbId may have decimals -> int
    links["tmdbId"] = links["tmdbId"].astype(int)
    return dict(zip(links["movieId"].tolist(), links["tmdbId"].tolist()))

movie_to_tmdb = load_links(LINKS_PATH)

# -------------------- TMDb helper (optional) --------------------
# Get key from Streamlit secrets or environment
TMDB_API_KEY = None
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    import os
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", None)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"  # small poster width

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fetch_tmdb_metadata(tmdb_id: int):
    """Return dict with 'poster_url' and 'overview' or None on failure."""
    if TMDB_API_KEY is None or not tmdb_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=5)
        resp.raise_for_status()
        d = resp.json()
        poster = d.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster}" if poster else None
        overview = d.get("overview") or ""
        title_tmdb = d.get("title") or None
        return {"poster_url": poster_url, "overview": overview, "title_tmdb": title_tmdb}
    except RequestException:
        return None

# -------------------- TF-IDF recommenders --------------------
def recommend_from_title(query: str, top_k: int = 10, min_sim: float = 0.10):
    if not query or not query.strip():
        return pd.DataFrame(columns=["movieId","title","genres"]), 0.0
    q = query.strip().lower()
    q_vec = tfidf_vec.transform([q])
    sims_to_query = cosine_similarity(q_vec, X).ravel()
    best_sim = float(np.max(sims_to_query)) if sims_to_query.size else 0.0
    if best_sim < min_sim:
        return pd.DataFrame(columns=["movieId","title","genres"]), best_sim
    seed_idx = int(np.argmax(sims_to_query))
    seed_vec = X[seed_idx]
    sims = cosine_similarity(seed_vec, X).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != seed_idx][:top_k]
    recs = movies.iloc[order][["movieId","title", "genres"]].reset_index(drop=True)
    recs.index = range(1, len(recs) + 1)
    return recs, best_sim

def recommend_from_likes_safe(selected_titles: list, top_k: int = 10):
    if not selected_titles:
        return pd.DataFrame(columns=["movieId","title","genres"])
    if not isinstance(selected_titles, (list, tuple)):
        raise TypeError("selected_titles must be a list or tuple of titles")
    mask = movies["title"].isin(selected_titles)
    idxs = movies.index[mask].tolist()
    if not idxs:
        return pd.DataFrame(columns=["movieId","title","genres"])
    profile = X[idxs].mean(axis=0)
    if sparse.issparse(profile):
        query = profile
    else:
        query = np.asarray(profile)
        if query.ndim == 1:
            query = query.reshape(1, -1)
    sims = cosine_similarity(query, X).ravel()
    order = np.argsort(-sims)
    liked_set = set(idxs)
    rec_indices = [i for i in order if i not in liked_set][:top_k]
    recs = movies.iloc[rec_indices][["movieId","title", "genres"]].copy().reset_index(drop=True)
    recs.index = range(1, len(recs) + 1)
    return recs

# -------------------- ALS artifacts loader --------------------
@st.cache_resource(show_spinner=False)
def load_als_artifacts():
    md = MODELS_DIR
    need = [md / "item_factors.npy", md / "user_factors.npy", md / "mappings.pkl", md / "train_user_seen.pkl"]
    for p in need:
        if not p.exists():
            return None
    item_f = np.load(md / "item_factors.npy")
    user_f = np.load(md / "user_factors.npy")
    with open(md / "mappings.pkl", "rb") as f:
        maps = pickle.load(f)
    with open(md / "train_user_seen.pkl", "rb") as f:
        train_user_seen = pickle.load(f)
    return {"item_f": item_f, "user_f": user_f, "maps": maps, "train_user_seen": train_user_seen}

als_art = load_als_artifacts()
if als_art is None:
    st.sidebar.warning("ALS artifacts not found in /models. ALS and Hybrid options will be disabled until artifacts are placed.")
else:
    item_f = als_art["item_f"]
    user_f = als_art["user_f"]
    maps = als_art["maps"]
    train_user_seen = als_art["train_user_seen"]
    item2idx = maps["item2idx"]
    idx2item = maps["idx2item"]
    user2idx = maps["user2idx"]
    idx2user = maps["idx2user"]
    # helper: map movieId -> row index in movies DataFrame (for TF-IDF alignment)
    movieid_to_row = {int(mid): int(i) for i, mid in movies["movieId"].reset_index(drop=True).items()}
    n_items = item_f.shape[0]

# -------------------- ALS / Hybrid recommenders --------------------
def als_recommend_for_user(raw_userid, k=10):
    if als_art is None:
        return pd.DataFrame(columns=["movieId","title","score"])
    if raw_userid not in user2idx:
        return pd.DataFrame(columns=["movieId","title","score"])
    uidx = user2idx[raw_userid]
    uvec = user_f[uidx]
    scores = item_f.dot(uvec)
    seen = train_user_seen.get(uidx, np.array([], dtype=int))
    if seen.size:
        seen = seen[seen < scores.shape[0]]
        scores[seen] = -np.inf
    if k >= scores.size:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        order = part[np.argsort(-scores[part])]
    rows = []
    for iid in order[:k]:
        mid = idx2item[int(iid)]
        title = movies.loc[movies['movieId']==mid, 'title'].iloc[0]
        rows.append({"movieId": int(mid),"title": title, "score": float(scores[iid])})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df

def hybrid_recommend(raw_userid, k=10, als_weight=0.6):
    if als_art is None:
        return pd.DataFrame(columns=["movieId","title","score"])
    if raw_userid not in user2idx:
        return pd.DataFrame(columns=["movieId","title","score"])
    uidx = user2idx[raw_userid]
    uvec = user_f[uidx]
    als_scores = item_f.dot(uvec).astype(float)
    seen_raw = [idx2item[i] for i in train_user_seen.get(uidx, [])]
    if len(seen_raw) == 0:
        tf_scores_items = np.zeros(als_scores.shape[0], dtype=float)
    else:
        mask = np.asarray(movies['movieId'].isin(seen_raw), dtype=bool)
        if mask.sum() == 0:
            tf_scores_items = np.zeros(als_scores.shape[0], dtype=float)
        else:
            profile = X[mask].mean(axis=0)
            if sparse.issparse(profile):
                try:
                    profile_arr = profile.toarray()
                except Exception:
                    profile_arr = np.asarray(profile)
            else:
                profile_arr = np.asarray(profile)
            profile_arr = np.atleast_2d(profile_arr)
            tf_scores_all = cosine_similarity(profile_arr, X).ravel()
            tf_scores_items = np.zeros(als_scores.shape[0], dtype=float)
            for i in range(als_scores.shape[0]):
                mid = idx2item[int(i)]
                row = movieid_to_row.get(int(mid), None)
                tf_scores_items[i] = float(tf_scores_all[row]) if row is not None else 0.0
    scaler = MinMaxScaler()
    try:
        als_norm = scaler.fit_transform(als_scores.reshape(-1,1)).ravel()
        tf_norm = scaler.fit_transform(tf_scores_items.reshape(-1,1)).ravel()
    except Exception:
        als_norm = als_scores
        tf_norm = tf_scores_items
    combined = als_weight * als_norm + (1-als_weight) * tf_norm
    seen = train_user_seen.get(uidx, np.array([], dtype=int))
    if seen.size:
        combined[seen] = -np.inf
    if k >= len(combined):
        order = np.argsort(-combined)
    else:
        part = np.argpartition(-combined, k)[:k]
        order = part[np.argsort(-combined[part])]
    rows = []
    for iid in order[:k]:
        mid = idx2item[int(iid)]
        title = movies.loc[movies['movieId']==mid,'title'].iloc[0]
        rows.append({"movieId": int(mid), "title": title, "score": float(combined[iid])})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df

# -------------------- UI & rendering (posters + summary) --------------------
def short_text(s: str, n=220):
    if not s:
        return ""
    s = s.strip()
    return (s if len(s) <= n else s[:n].rsplit(" ",1)[0] + " ...")

@st.cache_data(show_spinner=False, ttl=60*60*24)
def get_tmdb_for_movieid(movieid: int):
    """Return metadata dict for movieId (poster_url, overview) or None."""
    tmdb_id = movie_to_tmdb.get(int(movieid))
    if tmdb_id is None or TMDB_API_KEY is None:
        return None
    return fetch_tmdb_metadata(tmdb_id)

def render_recommendations_with_posters(df: pd.DataFrame, cols_per_row: int = 3):
    """
    df expected to have columns: movieId, title (optionally genres, score).
    Renders posters and a short summary. If poster not available, shows title + text.
    """
    if df.empty:
        st.info("No recommendations.")
        return
    # create grid
    for start in range(0, len(df), cols_per_row):
        slice_df = df.iloc[start:start+cols_per_row]
        cols = st.columns(len(slice_df))
        for c, (_, row) in zip(cols, slice_df.iterrows()):
            movieid = int(row.get("movieId", None))
            title = row.get("title", "")
            score = row.get("score", None)
            meta = None
            try:
                meta = get_tmdb_for_movieid(movieid)
            except Exception:
                meta = None
            if meta and meta.get("poster_url"):
                c.image(meta["poster_url"], use_column_width=True, caption=title)
                c.write("**" + title + "**")
                overview = meta.get("overview", "")
                if overview:
                    c.write(short_text(overview, n=240))
            else:
                # no poster: show a placeholder with text
                c.write("**" + title + "**")
                # try TMDB overview even if no poster
                if meta and meta.get("overview"):
                    c.write(short_text(meta.get("overview", ""), n=240))
                else:
                    # fallback: show genres if available
                    gr = movies.loc[movies['movieId']==movieid, 'genres']
                    if not gr.empty:
                        c.write(gr.iloc[0])
            if score is not None:
                c.caption(f"score: {score:.3f}")

# -------------------- Sidebar & main UI --------------------
with st.sidebar:
    st.header("Settings")
    algorithm = st.selectbox("Algorithm", ["TF-IDF (content)", "ALS (collaborative)", "Hybrid (ALS+TF-IDF)"], index=0)
    k = st.slider("Number of recommendations", min_value=3, max_value=30, value=9)

st.write("### How to use")
st.write("- **TF-IDF**: type a movie name (free-text) or pick multiple liked movies.")
st.write("- **ALS**: pick a user id from the dropdown (requires models/*).")
st.write("- **Hybrid**: combine ALS & TF-IDF (best of both).")

if algorithm == "TF-IDF (content)":
    mode = st.radio("Mode:", ["Single movie", "Multiple likes"])    
    if mode == "Single movie":
        with st.form("single_form"):
            q = st.text_input("Enter a movie you like (free-text)")
            submitted = st.form_submit_button("Recommend")
            if submitted:
                recs, sim = recommend_from_title(q, top_k=k)
                if recs.empty:
                    st.warning("No strong match found. Try different keywords or use Multiple likes.")
                else:
                    st.subheader("Recommendations (TF-IDF)")
                    render_recommendations_with_posters(recs, cols_per_row=3)
    else:
        picks = st.multiselect("Select your liked movies (exact titles)", movies['title'].tolist(), max_selections=10)
        if st.button("Recommend"):
            if not picks:
                st.warning("Please select one or more movies.")
            else:
                recs = recommend_from_likes_safe(picks, top_k=k)
                st.subheader("Recommendations (TF-IDF)")
                render_recommendations_with_posters(recs, cols_per_row=3)

elif algorithm == "ALS (collaborative)":
    if als_art is None:
        st.error("ALS artifacts not found in models/. Train ALS and save item_factors/user_factors/mappings/train_user_seen.")
    else:
        user_list = sorted(list(maps['user2idx'].keys()))
        sel = st.selectbox("Select user id (demo)", user_list[:200])
        if st.button("Recommend (ALS)"):
            recs = als_recommend_for_user(sel, k)
            if recs.empty:
                st.warning("No recommendations (user not found or no artifacts).")
            else:
                st.subheader("Recommendations (ALS)")
                render_recommendations_with_posters(recs, cols_per_row=3)

else:  # Hybrid
    if als_art is None:
        st.error("ALS artifacts not found in models/. Hybrid disabled.")
    else:
        user_list = sorted(list(maps['user2idx'].keys()))
        sel = st.selectbox("Select user id (demo)", user_list[:200])
        als_w = st.slider("ALS weight", 0.0, 1.0, 0.6)
        if st.button("Recommend (Hybrid)"):
            recs = hybrid_recommend(sel, k, als_w)
            if recs.empty:
                st.warning("No recommendations (user not found or no artifacts).")
            else:
                st.subheader("Recommendations (Hybrid)")
                render_recommendations_with_posters(recs, cols_per_row=3)

st.markdown("---")
st.caption("TF-IDF + ALS hybrid recommender — use TF-IDF for cold-start and ALS for personalization. Posters loaded from TMDb when API key and mapping are available.")
