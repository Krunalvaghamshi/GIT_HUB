import streamlit as st
import pandas as pd
import requests
import time
import sys
import os
import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# -------------------------
# Streamlit Movie Recommender
# Single-file app — professional UI, TMDB-backed, semantic recommendations
# -------------------------

# ---------- CONFIG ----------
USER_FILE = "users.csv"
RECOMMENDED_HISTORY_FILE = "recommendations_history.csv"
WATCHED_HISTORY_FILE = "watched_history.csv"
USER_PREFS_FILE = "user_preferences.csv"

# Use environment variable if set, otherwise fallback key
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "b6b01a81b37421bb3416b5580793aa01")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")

# ---------- HELPERS / I/O ----------

def load_data_or_create_empty(filepath, columns):
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath, encoding='utf-8')
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)


def save_df(df, path):
    df.to_csv(path, index=False, encoding='utf-8')


# ---------- CACHED FUNCTIONS ----------

@st.cache_data(show_spinner=False)
def get_tmdb_genres_and_languages():
    """Fetch TMDB genres and languages (cached)."""
    try:
        genre_data = requests.get(f"{TMDB_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US", timeout=10).json()
        genre_map = {g['id']: g['name'] for g in genre_data.get('genres', [])}
        reverse = {g['name']: g['id'] for g in genre_data.get('genres', [])}
    except Exception as e:
        st.error("Failed to fetch TMDB genres. Check your internet / API key.")
        return {}, {}, {}
    try:
        lang_data = requests.get(f"{TMDB_BASE_URL}/configuration/languages?api_key={TMDB_API_KEY}", timeout=10).json()
        language_map = {l['english_name']: l['iso_639_1'] for l in lang_data}
    except Exception as e:
        st.error("Failed to fetch TMDB languages. Check your internet.")
        return {}, {}, {}
    return genre_map, language_map, reverse


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return tokenizer, model


def get_embedding_from_model(tokenizer, model, text):
    """Get sentence embedding using transformer model."""
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        model_out = model(**encoded)
        emb = model_out.last_hidden_state.mean(dim=1).numpy()
    return emb


@st.cache_data(show_spinner=False)
def discover_tmdb_movies_cached(chosen_genres, chosen_langs, chosen_age, max_pages=6):
    """Discover TMDB movies based on filters (cached)."""
    genre_map, language_map, reverse = get_tmdb_genres_and_languages()
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US',
        'sort_by': 'vote_average.desc',
        'vote_count.gte': 100,
    }

    if chosen_genres:
        genre_ids = [str(reverse.get(g)) for g in chosen_genres if reverse.get(g) is not None]
        if genre_ids:
            params['with_genres'] = ','.join(genre_ids)
    if chosen_langs:
        lang_codes = [language_map.get(lang) for lang in chosen_langs if language_map.get(lang) is not None]
        if lang_codes:
            params['with_original_language'] = ','.join(lang_codes)

    curr_year = datetime.datetime.now().year
    release_date_param = None
    if chosen_age == "Published in the last 5 years.":
        release_date_param = f"{curr_year - 5}-01-01"
    elif chosen_age == "Published in the last 10 years.":
        release_date_param = f"{curr_year - 10}-01-01"
    elif chosen_age == "Published in the last 25 years.":
        release_date_param = f"{curr_year - 25}-01-01"
    if release_date_param:
        params['primary_release_date.gte'] = release_date_param

    all_movies = []
    for page in range(1, max_pages + 1):
        params['page'] = page
        try:
            response = requests.get(f"{TMDB_BASE_URL}/discover/movie", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get('results'):
                break
            for movie in data.get('results', []):
                if not movie.get('release_date'):
                    continue
                genre_names = [genre_map.get(i, 'Unknown') for i in movie.get('genre_ids', [])]
                lang_code = movie.get('original_language', 'en')
                full_language = {v: k for k, v in language_map.items()}.get(lang_code, lang_code)
                if lang_code == 'en':
                    industry = 'Hollywood'
                elif full_language in ['Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam']:
                    industry = full_language + ' Cinema'
                else:
                    industry = 'World Cinema'

                poster = movie.get('poster_path')
                poster_url = IMAGE_BASE + poster if poster else None

                movie_details = {
                    'id': int(movie['id']),
                    'title': movie.get('title') or movie.get('original_title'),
                    'genre': ', '.join(genre_names),
                    'language': full_language,
                    'industry': industry,
                    'year': int(movie['release_date'].split('-')[0]),
                    'description': movie.get('overview') or 'No description available.',
                    'rating': movie.get('vote_average', 0),
                    'age_rating': 'PG-13',
                    'cat': 'Critically Acclaimed' if movie.get('vote_count', 0) > 1000 else 'Popular',
                    'poster_url': poster_url
                }
                all_movies.append(movie_details)
        except Exception:
            continue
        time.sleep(0.2)

    df = pd.DataFrame(all_movies).drop_duplicates(subset=['id']).reset_index(drop=True)
    if df.empty:
        return df
    df = df[df['year'] != 0].dropna(subset=['description']).reset_index(drop=True)
    return df


# ---------- Recommendation logic ----------

def recommend_movie_from_candidates(df_candidates, user_profile, seen_movie_ids, tokenizer, model):
    user_emb = get_embedding_from_model(tokenizer, model, user_profile)
    df_unseen = df_candidates[~df_candidates['id'].isin(list(seen_movie_ids))]
    if df_unseen.empty:
        return None

    sims = []
    for _, row in df_unseen.iterrows():
        movie_text = f"{row['title']} {row['genre']} {row['language']} {row['industry']} {row['cat']} {row['description']}"
        movie_emb = get_embedding_from_model(tokenizer, model, movie_text)
        sim = cosine_similarity(user_emb, movie_emb)[0][0]
        sims.append((float(sim), int(row['id'])))

    sims.sort(reverse=True, key=lambda x: x[0])
    best_id = sims[0][1] if sims else None
    if best_id:
        return df_unseen[df_unseen['id'] == best_id].iloc[0]
    return None


# ---------- Logging helpers ----------

def log_user_preferences(user_id, genres, languages, age_filter, mood, rec_movie_id=None, rec_movie_title=None):
    df_pref = load_data_or_create_empty(USER_PREFS_FILE, [
        'user_id', 'timestamp', 'genres', 'languages', 'age_filter', 'mood',
        'recommended_movie_id', 'recommended_movie_title'
    ])
    pref_row = pd.DataFrame({
        'user_id': [user_id],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'genres': [', '.join(genres)],
        'languages': [', '.join(languages)],
        'age_filter': [age_filter],
        'mood': [mood],
        'recommended_movie_id': [rec_movie_id if rec_movie_id else ''],
        'recommended_movie_title': [rec_movie_title if rec_movie_title else '']
    })
    df_pref = pd.concat([df_pref, pref_row], ignore_index=True)
    save_df(df_pref, USER_PREFS_FILE)


def log_recommendation(user_id, movie_id):
    df_rec = load_data_or_create_empty(RECOMMENDED_HISTORY_FILE, ['user_id', 'movie_id', 'timestamp'])
    new_row = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    df_rec = pd.concat([df_rec, new_row], ignore_index=True)
    save_df(df_rec, RECOMMENDED_HISTORY_FILE)


def log_watched_movie(user_id, movie_id, title):
    df_watched = load_data_or_create_empty(WATCHED_HISTORY_FILE, ['user_id', 'movie_id', 'title', 'timestamp'])
    if not ((df_watched['user_id'] == user_id) & (df_watched['movie_id'] == movie_id)).any():
        new_row = pd.DataFrame({
            'user_id': [user_id],
            'movie_id': [movie_id],
            'title': [title],
            'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        df_watched = pd.concat([df_watched, new_row], ignore_index=True)
        save_df(df_watched, WATCHED_HISTORY_FILE)


# ---------- UI Layout ----------

def main_app():
    st.markdown("<style> .stApp {background: linear-gradient(120deg,#0f172a 0%, #001429 100%);} .card-title{font-weight:700;} </style>", unsafe_allow_html=True)

    st.title("🎯 AI Movie Recommender — Streamlit Edition")
    st.caption("Semantic recommendations + TMDB discovery. Professional UI with logging and history.")

    col1, col2 = st.columns([1, 3], gap='large')

    with col1:
        st.subheader("User")
        username = st.text_input("Enter username (create new or existing):", value=st.session_state.get('username', ''))
        if st.button("Login / Create"):
            if not username.strip():
                st.warning("Please enter a username.")
            else:
                df_users = load_data_or_create_empty(USER_FILE, ['user_id', 'username'])
                if username in df_users['username'].values:
                    user_id = int(df_users[df_users['username'] == username]['user_id'].iloc[0])
                    st.success(f"Welcome back, {username}!")
                else:
                    user_id = int(df_users['user_id'].max() + 1) if not df_users.empty else 1
                    df_users = pd.concat([df_users, pd.DataFrame({'user_id':[user_id], 'username':[username]})], ignore_index=True)
                    save_df(df_users, USER_FILE)
                    st.success(f"New user '{username}' created.")
                st.session_state['username'] = username
                st.session_state['user_id'] = user_id

        if 'user_id' in st.session_state:
            st.markdown(f"**Logged in:** {st.session_state['username']} (id: {st.session_state['user_id']})")

        st.divider()

        st.subheader("Filters")
        genre_map, language_map, reverse = get_tmdb_genres_and_languages()
        all_genres = sorted(list(reverse.keys()))
        avail_langs = sorted(list(language_map.keys()))

        chosen_genres = st.multiselect("Select genres", options=all_genres, default=all_genres[:3])
        default_lang = 'English' if 'English' in avail_langs else (avail_langs[0] if avail_langs else None)
        chosen_langs = st.multiselect("Select languages", options=avail_langs, default=[default_lang] if default_lang else [])
        age_options = ["Doesn’t matter.", "Published in the last 5 years.", "Published in the last 10 years.", "Published in the last 25 years."]
        chosen_age = st.selectbox("Age of movie", options=age_options, index=0)
        moods = ['Happy', 'Neutral', 'Sad']
        mood = st.selectbox("How are you today?", options=moods, index=1)

        if st.button("Get Recommendation", use_container_width=True):
            if 'user_id' not in st.session_state:
                st.warning('Please login or create a username first.')
            else:
                st.session_state['chosen_genres'] = chosen_genres
                st.session_state['chosen_langs'] = chosen_langs
                st.session_state['chosen_age'] = chosen_age
                st.session_state['mood'] = mood
                run_recommendation_flow()

        st.write("")
        st.expander("Quick actions")
        if st.button("Show my watched history"):
            if 'user_id' in st.session_state:
                df_w = load_data_or_create_empty(WATCHED_HISTORY_FILE, ['user_id', 'movie_id', 'title', 'timestamp'])
                user_w = df_w[df_w['user_id'] == st.session_state['user_id']]
                if user_w.empty:
                    st.info("No watched movies recorded yet.")
                else:
                    st.table(user_w.sort_values(by='timestamp', ascending=False).head(50))
            else:
                st.warning('Login to view history.')

    with col2:
        st.subheader("Recommendation Space")
        if 'last_recommendation' in st.session_state:
            rec = st.session_state['last_recommendation']
            display_movie_card(rec)
        else:
            st.info("No recommendation yet. Use the controls on the left to get started.")


# ---------- Run recommendation flow ----------

def run_recommendation_flow():
    tokenizer, model = load_embedding_model()
    user_id = st.session_state['user_id']
    chosen_genres = st.session_state.get('chosen_genres', [])
    chosen_langs = st.session_state.get('chosen_langs', [])
    chosen_age = st.session_state.get('chosen_age', "Doesn’t matter.")
    mood = st.session_state.get('mood', 'Neutral')

    with st.spinner('Discovering movies from TMDB...'):
        df_movies = discover_tmdb_movies_cached(tuple(chosen_genres), tuple(chosen_langs), chosen_age)

    if df_movies.empty:
        st.error('No movies matched your filters. Try different filters.')
        log_user_preferences(user_id, chosen_genres, chosen_langs, chosen_age, mood, rec_movie_id='', rec_movie_title='No movies found')
        return

    industries = sorted(df_movies['industry'].unique())
    user_profile = " ".join([mood, " ".join(chosen_genres), chosen_age, " ".join(industries), " ".join(chosen_langs)])

    df_watched = load_data_or_create_empty(WATCHED_HISTORY_FILE, ['user_id', 'movie_id', 'title', 'timestamp'])
    user_watched = df_watched[df_watched['user_id'] == user_id]
    seen_movie_ids = set(user_watched['movie_id'].astype('int').unique()) if not user_watched.empty else set()

    with st.spinner('Computing semantic recommendation...'):
        rec_movie = recommend_movie_from_candidates(df_movies, user_profile, seen_movie_ids, tokenizer, model)

    if rec_movie is None:
        st.warning('No unseen movie found with current filters.')
        log_user_preferences(user_id, chosen_genres, chosen_langs, chosen_age, mood, rec_movie_id='', rec_movie_title='No unseen movie')
        return

    log_user_preferences(user_id, chosen_genres, chosen_langs, chosen_age, mood, rec_movie_id=int(rec_movie['id']), rec_movie_title=rec_movie['title'])

    st.session_state['last_recommendation'] = rec_movie.to_dict()
    log_recommendation(user_id, int(rec_movie['id']))
    st.success(f"Recommended: {rec_movie['title']} — check the right panel")


# ---------- UI movie card ----------

def display_movie_card(rec):
    cols = st.columns([1, 2])
    with cols[0]:
        if rec.get('poster_url'):
            st.image(rec.get('poster_url'), use_column_width=True, caption=rec.get('title'))
        else:
            st.write('No poster available')
    with cols[1]:
        st.markdown(f"### {rec.get('title')} ({rec.get('year')})")
        st.markdown(f"**Genre:** {rec.get('genre')}  &nbsp; • &nbsp; **Language:** {rec.get('language')}")
        st.markdown(f"**Industry:** {rec.get('industry')}  &nbsp; • &nbsp; **Rating:** {rec.get('rating')}/10")
        st.markdown(f"**Tag:** {rec.get('cat')}  &nbsp; • &nbsp; **Censor:** {rec.get('age_rating')}")
        st.write(rec.get('description'))

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button('Mark as Watched'):
                if 'user_id' in st.session_state:
                    log_watched_movie(st.session_state['user_id'], int(rec.get('id')), rec.get('title'))
                    st.success('Marked as watched ✅')
                else:
                    st.warning('Please login first.')
        with c2:
            if st.button('I have watched this'):
                if 'user_id' in st.session_state:
                    log_watched_movie(st.session_state['user_id'], int(rec.get('id')), rec.get('title'))
                    st.info('Thanks — we will avoid recommending this again.')
                else:
                    st.warning('Please login first.')
        with c3:
            if st.button('Next Recommendation'):
                if 'user_id' in st.session_state:
                    st.session_state.pop('last_recommendation', None)
                    run_recommendation_flow()
                else:
                    st.warning('Login required')


# ---------- Entry point ----------

if __name__ == '__main__':
    main_app()
