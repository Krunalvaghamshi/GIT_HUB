import streamlit as st
import pandas as pd
import torch
import requests
import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ----------------- Constants & Globals -----------------
USER_FILE = 'users.csv'
RECOMMENDED_HISTORY_FILE = 'recommendations_history.csv'
WATCHED_HISTORY_FILE = 'watched_history.csv'
USER_PREFS_FILE = 'user_preferences.csv'

TMDB_API_KEY = "b6b01a81b37421bb3416b5580793aa01"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

GENRE_MAP, LANGUAGE_MAP, REVERSE_GENRE_MAP = {}, {}, {}

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# ----------------- Utility Functions -----------------

def load_csv(filepath, columns):
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except:
            return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame(columns=columns)

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)

def get_user_id(username):
    df = load_csv(USER_FILE, ['user_id', 'username'])
    user = df[df['username'] == username]
    if not user.empty:
        return user.iloc[0]['user_id']
    new_id = df['user_id'].max() + 1 if not df.empty else 1
    new_user = pd.DataFrame({'user_id': [new_id], 'username': [username]})
    df = pd.concat([df, new_user], ignore_index=True)
    save_csv(df, USER_FILE)
    return new_id

def log_user_preferences(user_id, genres, langs, age_filter, mood, rec_movie_id=None, rec_movie_title=None):
    df = load_csv(USER_PREFS_FILE, ['user_id', 'timestamp', 'genres', 'languages', 'age_filter', 'mood', 'recommended_movie_id', 'recommended_movie_title'])
    new_row = pd.DataFrame({
        'user_id': [user_id],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'genres': [", ".join(genres)],
        'languages': [", ".join(langs)],
        'age_filter': [age_filter],
        'mood': [mood],
        'recommended_movie_id': [rec_movie_id or ""],
        'recommended_movie_title': [rec_movie_title or ""]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    save_csv(df, USER_PREFS_FILE)

def log_recommendation(user_id, movie_id):
    df = load_csv(RECOMMENDED_HISTORY_FILE, ['user_id', 'movie_id', 'timestamp'])
    new_row = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    save_csv(df, RECOMMENDED_HISTORY_FILE)

def log_watched_movie(user_id, movie_id, title):
    df = load_csv(WATCHED_HISTORY_FILE, ['user_id', 'movie_id', 'title', 'timestamp'])
    if ((df['user_id'] == user_id) & (df['movie_id'] == movie_id)).any():
        return
    new_row = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id],
        'title': [title],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    save_csv(df, WATCHED_HISTORY_FILE)

def fetch_tmdb_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_tmdb_genres_and_languages():
    global GENRE_MAP, LANGUAGE_MAP, REVERSE_GENRE_MAP
    genres = fetch_tmdb_data(f"{TMDB_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US")
    if genres:
        GENRE_MAP = {g['id']: g['name'] for g in genres.get('genres', [])}
        REVERSE_GENRE_MAP = {v:k for k,v in GENRE_MAP.items()}
    langs = fetch_tmdb_data(f"{TMDB_BASE_URL}/configuration/languages?api_key={TMDB_API_KEY}")
    if langs:
        LANGUAGE_MAP = {l['english_name']: l['iso_639_1'] for l in langs}
    return bool(GENRE_MAP and LANGUAGE_MAP)

def discover_tmdb_movies(genres, langs, age_filter, max_pages=5):
    all_movies = []
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US',
        'sort_by': 'vote_average.desc',
        'vote_count.gte': 100,
    }
    if genres:
        genre_ids = [str(REVERSE_GENRE_MAP.get(g)) for g in genres if REVERSE_GENRE_MAP.get(g)]
        params['with_genres'] = ",".join(genre_ids)
    if langs:
        lang_codes = [LANGUAGE_MAP.get(lang) for lang in langs if LANGUAGE_MAP.get(lang)]
        params['with_original_language'] = ",".join(lang_codes)
    curr_year = datetime.datetime.now().year
    date_param = None
    if age_filter == "Published in the last 5 years.":
        date_param = f"{curr_year - 5}-01-01"
    elif age_filter == "Published in the last 10 years.":
        date_param = f"{curr_year - 10}-01-01"
    elif age_filter == "Published in the last 25 years.":
        date_param = f"{curr_year - 25}-01-01"
    if date_param:
        params['primary_release_date.gte'] = date_param

    for page in range(1, max_pages+1):
        params['page'] = page
        url = f"{TMDB_BASE_URL}/discover/movie"
        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('results', [])
            if not results:
                break
            for movie in results:
                lang_code = movie.get('original_language', 'en')
                full_language = {v:k for k,v in LANGUAGE_MAP.items()}.get(lang_code, lang_code)
                genre_names = [GENRE_MAP.get(i, 'Unknown') for i in movie.get('genre_ids',[])]
                if lang_code == 'en':
                    industry = 'Hollywood'
                elif full_language in ['Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam']:
                    industry = full_language + ' Cinema'
                else:
                    industry = 'World Cinema'
                release_date = movie.get('release_date')
                if not release_date:
                    continue
                movie_info = {
                    'id': movie['id'],
                    'title': movie['title'],
                    'genre': ", ".join(genre_names),
                    'language': full_language,
                    'industry': industry,
                    'year': int(release_date.split('-')[0]),
                    'description': movie.get('overview', 'No description.'),
                    'rating': movie.get('vote_average',0),
                    'age_rating': 'PG-13',
                    'cat': 'Critically Acclaimed' if movie.get('vote_count',0) > 1000 else 'Popular',
                }
                all_movies.append(movie_info)
        except:
            pass
        time.sleep(0.2)
    df = pd.DataFrame(all_movies)
    if df.empty:
        df = pd.DataFrame(columns=['id','title','genre','language','industry','year','description','rating','age_rating','cat'])
    return df

def get_embedding(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        emb = model(**encoded)
        return emb.last_hidden_state.mean(dim=1).numpy()

def recommend_movie(df, user_profile, seen_movie_ids):
    user_emb = get_embedding(user_profile)
    unseen_movies = df[~df['id'].isin(seen_movie_ids)]
    if unseen_movies.empty:
        return None
    sims = []
    for _, row in unseen_movies.iterrows():
        movie_text = f"{row['title']} {row['genre']} {row['language']} {row['industry']} {row['cat']} {row['description']}"
        movie_emb = get_embedding(movie_text)
        sim = cosine_similarity(user_emb, movie_emb)[0][0]
        sims.append((sim, row['id']))
    sims.sort(reverse=True)
    best_id = sims[0][1]
    return unseen_movies[unseen_movies['id'] == best_id].iloc[0]

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="centered")

# Load TMDB data on app start
if 'tmdb_loaded' not in st.session_state:
    with st.spinner("Loading genres and languages from TMDB..."):
        if not get_tmdb_genres_and_languages():
            st.error("Failed to load TMDB data. Check your network and API key.")
            st.stop()
    st.session_state.tmdb_loaded = True

# User Authentication sidebar
st.sidebar.header("User Authentication")
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.authenticated:
    auth_choice = st.sidebar.radio("Are you a:", ("New User", "Existing User"))
    username_input = st.sidebar.text_input("Username")
    if st.sidebar.button("Proceed"):
        if not username_input:
            st.sidebar.warning("Please enter a username.")
        else:
            if auth_choice == "New User":
                df_users = load_csv(USER_FILE, ['user_id', 'username'])
                if username_input in df_users['username'].values:
                    st.sidebar.error("Username already exists. Choose another.")
                else:
                    uid = get_user_id(username_input)
                    st.session_state.username = username_input
                    st.session_state.user_id = uid
                    st.session_state.authenticated = True
                    st.experimental_rerun()
            else:  # Existing User
                df_users = load_csv(USER_FILE, ['user_id', 'username'])
                if username_input not in df_users['username'].values:
                    st.sidebar.error("Username not found.")
                else:
                    uid = get_user_id(username_input)
                    st.session_state.username = username_input
                    st.session_state.user_id = uid
                    st.session_state.authenticated = True
                    st.experimental_rerun()
else:
    st.sidebar.markdown(f"### Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# -- Main application --

if st.session_state.authenticated:

    st.title("🎬 AI Movie Recommender")

    # Load watched movies
    df_watched = load_csv(WATCHED_HISTORY_FILE, ['user_id', 'movie_id', 'title', 'timestamp'])
    watched_ids = set(df_watched[df_watched['user_id'] == st.session_state.user_id]['movie_id'].unique())

    # Persistent session state for filters
    if 'filters_set' not in st.session_state:
        st.session_state.filters_set = False
        st.session_state.genres = []
        st.session_state.langs = []
        st.session_state.age = "Doesn’t matter."
        st.session_state.mood = "Neutral"
        st.session_state.seen_movies = watched_ids.copy()
        st.session_state.last_recommended = None

    def reset_filters():
        st.session_state.filters_set = False

    if not st.session_state.filters_set or st.button("Edit Filters"):
        st.session_state.genres = st.multiselect(
            "Choose genres",
            options=sorted(REVERSE_GENRE_MAP.keys()),
            default=st.session_state.genres)
        st.session_state.langs = st.multiselect(
            "Choose languages",
            options=sorted(LANGUAGE_MAP.keys()),
            default=st.session_state.langs)
        st.session_state.age = st.selectbox(
            "How old should movies be?",
            ["Doesn’t matter.", "Published in the last 5 years.", "Published in the last 10 years.", "Published in the last 25 years."],
            index=["Doesn’t matter.", "Published in the last 5 years.", "Published in the last 10 years.", "Published in the last 25 years."].index(st.session_state.age))
        st.session_state.mood = st.selectbox(
            "How are you today?",
            options=['Happy', 'Neutral', 'Sad'],
            index=['Happy', 'Neutral', 'Sad'].index(st.session_state.mood))

        if st.button("Apply Filters"):
            st.session_state.filters_set = True
            st.session_state.seen_movies = watched_ids.copy()
            st.session_state.last_recommended = None

    if st.session_state.filters_set:
        with st.spinner("Fetching and recommending movies..."):
            movies_df = discover_tmdb_movies(st.session_state.genres, st.session_state.langs, st.session_state.age)
            if movies_df.empty:
                st.warning("No movies found with these filters. Try changing filters.")
            else:
                user_prof = " ".join([st.session_state.mood, ", ".join(st.session_state.genres), st.session_state.age,
                                     ", ".join(sorted(movies_df['industry'].unique())), ", ".join(st.session_state.langs)])

                rec_movie = recommend_movie(movies_df, user_prof, st.session_state.seen_movies)
                if rec_movie is None:
                    st.info("No new unseen movies found under current filters.")
                else:
                    st.session_state.last_recommended = rec_movie
                    log_user_preferences(st.session_state.user_id, st.session_state.genres, st.session_state.langs, st.session_state.age, st.session_state.mood,
                                         rec_movie_id=rec_movie['id'], rec_movie_title=rec_movie['title'])
                    log_recommendation(st.session_state.user_id, rec_movie['id'])

                    st.markdown(f"### 🎞️ {rec_movie['title']}  ")
                    st.markdown(f"**Genre:** {rec_movie['genre']}  ")
                    st.markdown(f"**Language:** {rec_movie['language']}  ")
                    st.markdown(f"**Industry:** {rec_movie['industry']}  ")
                    st.markdown(f"**Year:** {rec_movie['year']}  ")
                    st.markdown(f"**TMDB Rating:** {rec_movie['rating']} / 10  ")
                    st.markdown(f"**Certification:** {rec_movie['age_rating']}  ")
                    st.markdown(f"**Tag:** {rec_movie['cat']}  ")
                    with st.expander("Description"):
                        st.write(rec_movie['description'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Yes, I've watched this"):
                            log_watched_movie(st.session_state.user_id, rec_movie['id'], rec_movie['title'])
                            st.session_state.seen_movies.add(rec_movie['id'])
                            st.experimental_rerun()
                    with col2:
                        if st.button("No, but I will watch it now"):
                            log_watched_movie(st.session_state.user_id, rec_movie['id'], rec_movie['title'])
                            st.session_state.seen_movies.add(rec_movie['id'])
                            st.success("Added to your watched history. Enjoy your movie! 🎉")
                            st.experimental_rerun()
                    with col3:
                        if st.button("No, I haven't watched this"):
                            st.info("Enjoy your movie!")
                            st.experimental_rerun()

    # Show watched history summary
    st.sidebar.header("Watched History")
    user_watched_df = df_watched[df_watched['user_id'] == st.session_state.user_id]
    if not user_watched_df.empty:
        st.sidebar.dataframe(user_watched_df[['title', 'timestamp']].sort_values(by='timestamp', ascending=False))
    else:
        st.sidebar.write("No movies watched yet.")

    # Show recommendation and preference logs in sidebar (optional)
    if st.sidebar.checkbox("Show Preferences & Recommendations Log"):
        prefs_df = load_csv(USER_PREFS_FILE, ['user_id', 'timestamp', 'genres', 'languages', 'age_filter', 'mood', 'recommended_movie_id', 'recommended_movie_title'])
        prefs_user = prefs_df[prefs_df['user_id'] == st.session_state.user_id]
        st.sidebar.dataframe(prefs_user.sort_values(by='timestamp', ascending=False))

        rec_df = load_csv(RECOMMENDED_HISTORY_FILE, ['user_id', 'movie_id', 'timestamp'])
        rec_user = rec_df[rec_df['user_id'] == st.session_state.user_id]
        st.sidebar.dataframe(rec_user.sort_values(by='timestamp', ascending=False))

