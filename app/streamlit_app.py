"""
Streamlit app (fixed): use TMDb when available, else use Wikimedia REST summary endpoint for posters+extract.
This file is a focused patch over previous app: improved Wikipedia fetching and visible debug source badges.
"""
from pathlib import Path
import pickle, urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import requests
from requests.exceptions import RequestException
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-latest-small"
MODELS_DIR = ROOT / "models"
LINKS_PATH = DATA_DIR / "links.csv"

st.set_page_config(page_title="Movie Recommender (TF-IDF + ALS)", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender — Posters (TMDb → Wikipedia REST fallback)")

# --- load movies (unchanged) ---
@st.cache_data(show_spinner=False)
def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"movieId", "title", "genres"}.issubset(df.columns):
        raise ValueError("movies.csv must contain columns: movieId,title,genres")
    df = df.copy()
    df["genres"] = df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    df["clean_title"] = df["title"].astype(str).str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float).fillna(np.nan).astype("Int64")
    df["doc"] = (df["clean_title"].fillna("") + " " + df["genres"]).str.lower()
    return df[["movieId","title","clean_title","year","genres","doc"]].reset_index(drop=True)

movies = load_movies(DATA_DIR / "movies.csv")

@st.cache_resource(show_spinner=False)
def build_tfidf(docs: pd.Series):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(docs)
    return vec, X

tfidf_vec, X = build_tfidf(movies["doc"])

# --- load links mapping ---
@st.cache_data(show_spinner=False)
def load_links(path: Path):
    if not path.exists():
        return {}
    links = pd.read_csv(path)
    if "tmdbId" in links.columns:
        links = links[["movieId","tmdbId"]].dropna()
        links["movieId"] = links["movieId"].astype(int)
        links["tmdbId"] = links["tmdbId"].astype(int)
        return dict(zip(links["movieId"].tolist(), links["tmdbId"].tolist()))
    return {}

movie_to_tmdb = load_links(LINKS_PATH)

# --- TMDb optional (unchanged) ---
TMDB_API_KEY = None
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    import os
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", None)
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

def fetch_tmdb_metadata(tmdb_id: int):
    if TMDB_API_KEY is None or tmdb_id is None:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=6)
        resp.raise_for_status()
        d = resp.json()
        poster = d.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster}" if poster else None
        overview = d.get("overview") or ""
        return {"poster_url": poster_url, "overview": overview, "source": "tmdb"}
    except RequestException:
        return None

# --- NEW: Wikimedia REST summary endpoint (reliable) ---
WIKI_REST_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"

def fetch_wikipedia_summary(title: str, year=None):
    """
    Use Wikimedia REST summary endpoint. Returns dict {poster_url, overview, source} or None.
    """
    if not title:
        return None
    # prefer "Title (year)" when year is available to disambiguate
    query_title = title
    if year and not pd.isna(year):
        try:
            y = int(year)
            query_title = f"{title} ({y})"
        except Exception:
            query_title = title
    # URL-encode the page title; use requests to fetch
    encoded = urllib.parse.quote(query_title, safe='')
    url = WIKI_REST_BASE + encoded
    headers = {"User-Agent": "MovieRecommender/1.0 (contact: you@example.com)"}  # replace contact if you like
    try:
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 404:
            # fallback: try searching without year
            if "(" in query_title:
                return fetch_wikipedia_summary(title, year=None)
            return None
        r.raise_for_status()
        js = r.json()
        thumb = None
        if "thumbnail" in js and js["thumbnail"] and js["thumbnail"].get("source"):
            thumb = js["thumbnail"]["source"]
        extract = js.get("extract") or ""
        return {"poster_url": thumb, "overview": extract, "source": "wiki"}
    except RequestException:
        return None
    except ValueError:
        return None

# --- unified metadata getter with caching ---
@st.cache_data(show_spinner=False, ttl=60*60*24)
def get_metadata_for_movie(movieid: int, title: str, year):
    # 1) TMDb if available
    tmdb_id = movie_to_tmdb.get(int(movieid))
    if tmdb_id is not None and TMDB_API_KEY:
        m = fetch_tmdb_metadata(tmdb_id)
        if m:
            return m
    # 2) Wikimedia REST fallback
    wiki_m = fetch_wikipedia_summary(title, year)
    if wiki_m:
        return wiki_m
    # 3) none
    return {"poster_url": None, "overview": "", "source": "none"}

# --- helper for rendering cards ---
def short_text(s: str, n=240):
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n].rsplit(" ",1)[0] + " ..."

def render_cards_from_df(df, cols_per_row=3):
    if df.empty:
        st.info("No recommendations.")
        return
    for start in range(0, len(df), cols_per_row):
        slice_df = df.iloc[start:start+cols_per_row]
        cols = st.columns(len(slice_df))
        for c, (_, row) in zip(cols, slice_df.iterrows()):
            mid = int(row.get("movieId"))
            title = row.get("title", "")
            score = row.get("score", None)
            # fetch metadata (cached)
            meta = get_metadata_for_movie(mid, movies.loc[movies['movieId']==mid, 'clean_title'].iloc[0],
                                         movies.loc[movies['movieId']==mid, 'year'].iloc[0])
            poster = meta.get("poster_url")
            overview = meta.get("overview")
            source = meta.get("source")
            if poster:
                # Show poster, then title and short summary
                try:
                    c.image(poster, use_column_width=True, caption=title)
                except Exception:
                    # image fetch/display error -> fallback to text
                    c.write("**" + title + "**")
                    if overview:
                        c.write(short_text(overview))
                else:
                    if overview:
                        c.write(short_text(overview))
            else:
                # text fallback
                c.write("**" + title + "**")
                if overview:
                    c.write(short_text(overview))
                else:
                    # show genres as fallback
                    gr = movies.loc[movies['movieId']==mid, 'genres']
                    if not gr.empty:
                        c.write(gr.iloc[0])
            if score is not None:
                c.caption(f"score: {score:.3f} • meta: {source}")

# --- minimal TF-IDF & ALS & Hybrid (kept simple) ---
# (Re-use your existing recommenders; assume they exist below)
# For brevity include a minimal TF-IDF recommender and reuse ALS/hybrid functions if present.
# If your project file already contains recommend_from_title, recommend_from_likes_safe,
# als_recommend_for_user and hybrid_recommend, keep them. Otherwise a small TF-IDF sample:
def recommend_from_title(q, top_k=9):
    q = q or ""
    q_vec = tfidf_vec.transform([q.strip().lower()])
    sims = cosine_similarity(q_vec, X).ravel()
    if sims.size == 0:
        return pd.DataFrame(columns=["movieId","title"])
    order = np.argsort(-sims)[:top_k]
    recs = movies.iloc[order][["movieId","title"]].copy().reset_index(drop=True)
    return recs

# Try to load ALS artifacts (if present)
def load_als_artifacts_simple():
    md = MODELS_DIR
    need = [md/"item_factors.npy", md/"user_factors.npy", md/"mappings.pkl", md/"train_user_seen.pkl"]
    for p in need:
        if not p.exists():
            return None
    item_f = np.load(md/"item_factors.npy")
    user_f = np.load(md/"user_factors.npy")
    with open(md/"mappings.pkl","rb") as f:
        maps = pickle.load(f)
    with open(md/"train_user_seen.pkl","rb") as f:
        train_user_seen = pickle.load(f)
    return {"item_f":item_f,"user_f":user_f,"maps":maps,"train_user_seen":train_user_seen}

als_art = load_als_artifacts_simple()
has_als = als_art is not None
if has_als:
    item_f = als_art["item_f"]; user_f = als_art["user_f"]; maps = als_art["maps"]; train_user_seen = als_art["train_user_seen"]
    item2idx = maps["item2idx"]; idx2item = maps["idx2item"]; user2idx = maps["user2idx"]

# --- UI: simple demo to check posters ---
st.sidebar.header("Quick poster test")
test_title = st.sidebar.text_input("Movie title to test (e.g., Toy Story)", value="")
test_year = st.sidebar.text_input("Year (optional)", value="")
if st.sidebar.button("Fetch test metadata"):
    # find movieId candidates by title substring
    candidates = movies[movies['clean_title'].str.contains(test_title, case=False, na=False)]
    if candidates.empty:
        st.sidebar.warning("No local movie matched that title. Try a different query.")
    else:
        # pick first candidate, show metadata
        mid = int(candidates.iloc[0]['movieId'])
        ctitle = candidates.iloc[0]['clean_title']
        cyear = candidates.iloc[0]['year']
        st.sidebar.write("Local match:", ctitle, cyear, "movieId:", mid)
        meta = get_metadata_for_movie(mid, ctitle, cyear)
        st.sidebar.write("metadata (debug):", meta)
        if meta.get("poster_url"):
            st.sidebar.image(meta.get("poster_url"), width=150)
        else:
            st.sidebar.write("No poster found; source:", meta.get("source"))

st.write("### Demo recommendations (TF-IDF quick mode)")
q = st.text_input("Type a keyword or movie (quick TF-IDF demo):", value="Toy Story")
if st.button("Recommend (quick)"):
    recs = recommend_from_title(q, top_k=9)
    render_cards_from_df(recs, cols_per_row=3)

st.markdown("---")
st.caption("If posters are still not showing: check the small sidebar test (enter a title) — it prints metadata and displays the poster if found. If metadata shows 'source: none' paste the title and I'll debug that page.")
