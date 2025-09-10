"""
Streamlit app: TF-IDF content recommender + ALS collaborative + Hybrid
Place MovieLens ml-latest-small under data/ml-latest-small/
Place trained ALS artifacts under models/: item_factors.npy, user_factors.npy, mappings.pkl, train_user_seen.pkl
Run: streamlit run app/streamlit_app.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-latest-small"
MODELS_DIR = ROOT / "models"

# -------------------- Page config --------------------
st.set_page_config(page_title="Movie Recommender (TF-IDF + ALS)", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender — TF-IDF + ALS (Hybrid)")

# -------------------- Helpers: find files --------------------
def require_file(p: Path, what: str):
    if not p.exists():
        st.error(f"Missing {what}: {p}\nPlease put the required file and reload the app.")
        st.stop()

require_file(DATA_DIR / "movies.csv", "movies.csv")

# -------------------- Load & prepare movies + TF-IDF --------------------
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

# -------------------- TF-IDF recommenders --------------------
def recommend_from_title(query: str, top_k: int = 10, min_sim: float = 0.10):
    if not query or not query.strip():
        return pd.DataFrame(columns=["title", "genres"]), 0.0
    q = query.strip().lower()
    q_vec = tfidf_vec.transform([q])
    sims_to_query = cosine_similarity(q_vec, X).ravel()
    best_sim = float(np.max(sims_to_query)) if sims_to_query.size else 0.0
    if best_sim < min_sim:
        return pd.DataFrame(columns=["title", "genres"]), best_sim
    seed_idx = int(np.argmax(sims_to_query))
    seed_vec = X[seed_idx]
    sims = cosine_similarity(seed_vec, X).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != seed_idx][:top_k]
    recs = movies.iloc[order][["title", "genres"]].reset_index(drop=True)
    recs.index = range(1, len(recs) + 1)
    return recs, best_sim

def recommend_from_likes_safe(selected_titles: list, top_k: int = 10):
    if not selected_titles:
        return pd.DataFrame(columns=["title", "genres"])
    if not isinstance(selected_titles, (list, tuple)):
        raise TypeError("selected_titles must be a list or tuple of titles")
    mask = movies["title"].isin(selected_titles)
    idxs = movies.index[mask].tolist()
    if not idxs:
        return pd.DataFrame(columns=["title", "genres"])
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
    recs = movies.iloc[rec_indices][["title", "genres"]].copy().reset_index(drop=True)
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

# -------------------- ALS / Hybrid recommenders --------------------
def als_recommend_for_user(raw_userid, k=10):
    # returns DataFrame of title & score
    if als_art is None:
        return pd.DataFrame(columns=["title", "score"])
    if raw_userid not in user2idx:
        return pd.DataFrame(columns=["title", "score"])    
    uidx = user2idx[raw_userid]
    uvec = user_f[uidx]
    scores = item_f.dot(uvec)
    # mask seen
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
        rows.append({"title": title, "score": float(scores[iid])})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df

def hybrid_recommend(raw_userid, k=10, als_weight=0.6):
    # Hybrid: normalized ALS score + TF-IDF profile similarity
    if als_art is None:
        return pd.DataFrame(columns=["title","score"])
    if raw_userid not in user2idx:
        return pd.DataFrame(columns=["title","score"])    
    uidx = user2idx[raw_userid]
    # ALS scores
    uvec = user_f[uidx]
    als_scores = item_f.dot(uvec).astype(float)
    # TF-IDF profile from user's seen movies
    seen_raw = [idx2item[i] for i in train_user_seen.get(uidx, [])]
    if len(seen_raw) == 0:
        tf_scores = np.zeros(len(movies))
    else:
        # Convert pandas boolean mask to numpy boolean array to index sparse X safely
        mask_series = movies['movieId'].isin(seen_raw)
        mask = np.asarray(mask_series, dtype=bool)
        if mask.sum() == 0:
            tf_scores = np.zeros(len(movies))
        else:
            profile = X[mask].mean(axis=0)
            # ensure profile is 2D for cosine_similarity
            if sparse.issparse(profile):
                if getattr(profile, "ndim", 2) == 1:
                    profile = profile.reshape(1, -1)
            else:
                profile = np.atleast_2d(profile)
            tf_scores = cosine_similarity(profile, X).ravel()
    # normalize
    scaler = MinMaxScaler()
    try:
        als_norm = scaler.fit_transform(als_scores.reshape(-1,1)).ravel()
        tf_norm = scaler.fit_transform(tf_scores.reshape(-1,1)).ravel()
    except Exception:
        # fallback if zero-vector or any issue
        als_norm = als_scores
        tf_norm = tf_scores
    combined = als_weight * als_norm + (1-als_weight) * tf_norm
    # mask seen
    seen = train_user_seen.get(uidx, np.array([], dtype=int))
    if seen.size:
        combined[seen] = -np.inf
    # pick top
    if k >= len(combined):
        order = np.argsort(-combined)
    else:
        part = np.argpartition(-combined, k)[:k]
        order = part[np.argsort(-combined[part])]
    rows = []
    for iid in order[:k]:
        mid = idx2item[int(iid)]
        title = movies.loc[movies['movieId']==mid,'title'].iloc[0]
        rows.append({"title": title, "score": float(combined[iid])})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df

# -------------------- UI --------------------
with st.sidebar:
    st.header("Settings")
    algorithm = st.selectbox("Algorithm", ["TF-IDF (content)", "ALS (collaborative)", "Hybrid (ALS+TF-IDF)"], index=0)
    k = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)

st.write("### How to use")
st.write("- **TF-IDF**: type a movie name (free-text) or pick multiple liked movies.\n- **ALS**: pick a user id from the dropdown. Requires models/*.\n- **Hybrid**: combine ALS & TF-IDF (best of both worlds).")

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
                    st.dataframe(recs)
    else:
        picks = st.multiselect("Select your liked movies (exact titles)", movies['title'].tolist(), max_selections=10)
        if st.button("Recommend"):
            if not picks:
                st.warning("Please select one or more movies.")
            else:
                recs = recommend_from_likes_safe(picks, top_k=k)
                st.subheader("Recommendations (TF-IDF)")
                st.dataframe(recs)

elif algorithm == "ALS (collaborative)":
    if als_art is None:
        st.error("ALS artifacts not found in models/. Train ALS and save item_factors/user_factors/mappings/train_user_seen.")
    else:
        # limit dropdown length for UI
        user_list = sorted(list(maps['user2idx'].keys()))
        sel = st.selectbox("Select user id (demo)", user_list[:200])
        if st.button("Recommend (ALS)"):
            recs = als_recommend_for_user(sel, k)
            if recs.empty:
                st.warning("No recommendations (user not found or no artifacts).")
            else:
                st.subheader("Recommendations (ALS)")
                st.dataframe(recs)

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
                st.dataframe(recs)

st.markdown("---")
st.caption("TF-IDF + ALS hybrid recommender — use TF-IDF for cold-start and ALS for personalization.")
