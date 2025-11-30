import streamlit as st
import pandas as pd
import requests
import datetime
import torch
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="CineMind AI | Semantic Movie Guide",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Vision-Full" Look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: 700;
        color: #00d2ff;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    .subtitle {
        font-size: 20px !important;
        font-weight: 300;
        text-align: center;
        margin-bottom: 30px;
        color: #e0e0e0;
    }
    .movie-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        border: none;
        border-radius: 25px;
        height: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(221, 36, 118, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONSTANTS & SETUP ---
TMDB_API_KEY = "b6b01a81b37421bb3416b5580793aa01" # Ideally use st.secrets for this
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# --- 3. GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def get_google_sheet_client():
    # Only run this if secrets are present (prevents crash on local dev without secrets)
    if "gcp_service_account" not in st.secrets:
        st.warning("⚠️ Google Sheets secrets not found. Data will not be saved.")
        return None
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    client = gspread.authorize(creds)
    return client

def load_data_from_sheet(worksheet_name):
    client = get_google_sheet_client()
    if not client: return pd.DataFrame()
    try:
        sheet = client.open("Movie_Recommender_DB")
        worksheet = sheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def append_to_sheet(worksheet_name, row_dict):
    client = get_google_sheet_client()
    if not client: return
    try:
        sheet = client.open("Movie_Recommender_DB")
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.append_row([str(x) for x in row_dict.values()])
    except Exception as e:
        st.error(f"Sync Error: {e}")

# --- 4. AI MODEL LOADING (Cached) ---
@st.cache_resource
def load_ai_model():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return tokenizer, model

tokenizer, model = load_ai_model()

# --- 5. LOGIC FUNCTIONS (Adapted from your Code) ---
def get_embedding(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        model_out = model(**encoded)
        return model_out.last_hidden_state.mean(dim=1).numpy()

@st.cache_data
def get_tmdb_config():
    try:
        g_resp = requests.get(f"{TMDB_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US").json()
        l_resp = requests.get(f"{TMDB_BASE_URL}/configuration/languages?api_key={TMDB_API_KEY}").json()
        
        g_map = {g['id']: g['name'] for g in g_resp.get('genres', [])}
        rev_g_map = {g['name']: g['id'] for g in g_resp.get('genres', [])}
        l_map = {l['english_name']: l['iso_639_1'] for l in l_resp}
        return g_map, rev_g_map, l_map
    except:
        return {}, {}, {}

GENRE_MAP, REVERSE_GENRE_MAP, LANGUAGE_MAP = get_tmdb_config()

def discover_movies(chosen_genres, chosen_langs, chosen_age):
    params = {
        'api_key': TMDB_API_KEY, 'language': 'en-US', 
        'sort_by': 'vote_average.desc', 'vote_count.gte': 100, 'page': 1
    }
    
    if chosen_genres:
        g_ids = [str(REVERSE_GENRE_MAP.get(g)) for g in chosen_genres if g in REVERSE_GENRE_MAP]
        if g_ids: params['with_genres'] = ','.join(g_ids)
    
    if chosen_langs:
        l_codes = [LANGUAGE_MAP.get(l) for l in chosen_langs if l in LANGUAGE_MAP]
        if l_codes: params['with_original_language'] = '|'.join(l_codes)

    curr_year = datetime.datetime.now().year
    if "5 years" in chosen_age: params['primary_release_date.gte'] = f"{curr_year - 5}-01-01"
    elif "10 years" in chosen_age: params['primary_release_date.gte'] = f"{curr_year - 10}-01-01"
    elif "25 years" in chosen_age: params['primary_release_date.gte'] = f"{curr_year - 25}-01-01"

    all_movies = []
    # Fetch top 2 pages for speed/variety
    for page in range(1, 3):
        params['page'] = page
        try:
            data = requests.get(f"{TMDB_BASE_URL}/discover/movie", params=params).json()
            for m in data.get('results', []):
                all_movies.append({
                    'id': m['id'], 'title': m['title'], 'overview': m.get('overview', ''),
                    'rating': m.get('vote_average'), 'release_date': m.get('release_date', 'Unknown'),
                    'poster_path': m.get('poster_path'),
                    'genre_ids': m.get('genre_ids', []),
                    'full_text': f"{m['title']} {m.get('overview','')} {m.get('release_date','')}"
                })
        except: pass
    
    return pd.DataFrame(all_movies).drop_duplicates(subset=['id'])

def get_recommendation(df, user_profile, seen_ids):
    user_emb = get_embedding(user_profile)
    df_unseen = df[~df['id'].isin(seen_ids)]
    
    if df_unseen.empty: return None, 0.0

    # Batch embedding for speed
    movie_embs = get_embedding(df_unseen['full_text'].tolist())
    sims = cosine_similarity(user_emb, movie_embs)[0]
    
    df_unseen = df_unseen.copy()
    df_unseen['similarity'] = sims
    best_movie = df_unseen.sort_values(by='similarity', ascending=False).iloc[0]
    return best_movie, best_movie['similarity']

# --- 6. USER AUTHENTICATION ---
def authenticate():
    st.sidebar.markdown("### 👤 User Access")
    auth_mode = st.sidebar.radio("I am a:", ["New User", "Returning User"])
    
    username = st.sidebar.text_input("Enter Username")
    
    if st.sidebar.button("Login / Register"):
        if not username:
            st.sidebar.error("Please enter a username.")
            return

        with st.spinner("Connecting to User Database..."):
            df_users = load_data_from_sheet('users')
            
            user_id = None
            if not df_users.empty and 'username' in df_users.columns:
                existing = df_users[df_users['username'] == username]
                if not existing.empty:
                    user_id = int(existing.iloc[0]['user_id'])
            
            if auth_mode == "Returning User":
                if user_id:
                    st.session_state['user_id'] = user_id
                    st.session_state['username'] = username
                    st.sidebar.success(f"Welcome back, {username}!")
                    # Load history
                    df_watched = load_data_from_sheet('watched_history')
                    if not df_watched.empty:
                        user_watched = df_watched[df_watched['user_id'] == user_id]
                        st.session_state['seen_ids'] = set(user_watched['movie_id'].tolist())
                    else:
                        st.session_state['seen_ids'] = set()
                else:
                    st.sidebar.error("Username not found.")
            
            else: # New User
                if user_id:
                    st.sidebar.error("Username taken. Choose another.")
                else:
                    new_id = 1
                    if not df_users.empty and 'user_id' in df_users.columns:
                        new_id = int(df_users['user_id'].max()) + 1
                    
                    append_to_sheet('users', {'user_id': new_id, 'username': username})
                    st.session_state['user_id'] = new_id
                    st.session_state['username'] = username
                    st.session_state['seen_ids'] = set()
                    st.sidebar.success(f"Account created for {username}!")

# --- 7. MAIN APP LAYOUT ---
def main():
    st.markdown('<div class="big-font">CineMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Semantic Search • Neural Embeddings • Personalized Logic</div>', unsafe_allow_html=True)

    if 'user_id' not in st.session_state:
        authenticate()
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h3>👋 Please Login via the Sidebar to Begin</h3>
            <p>Access your history and get personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- User Logged In ---
    st.write(f"Logged in as: **{st.session_state['username']}** | Watched: **{len(st.session_state.get('seen_ids', []))}** movies")
    
    with st.expander("🛠️ Preference Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            genres = st.multiselect("Genres", options=sorted(REVERSE_GENRE_MAP.keys()))
        with c2:
            langs = st.multiselect("Languages", options=sorted(LANGUAGE_MAP.keys()), default=["English"])
        with c3:
            age = st.selectbox("Era", ["Doesn’t matter.", "Last 5 years", "Last 10 years", "Last 25 years"])
        
        mood = st.text_input("How are you feeling right now? (e.g., 'I want something inspiring but tragic')", placeholder="Type your mood here...")

    if st.button("🔮 Generate Recommendation"):
        if not mood and not genres:
            st.warning("Please select at least a genre or type a mood.")
        else:
            with st.status("🧠 Processing...", expanded=True) as status:
                st.write("Fetching movies from TMDB API...")
                df_candidates = discover_movies(genres, langs, age)
                
                if df_candidates.empty:
                    status.update(label="No movies found!", state="error")
                    st.error("No movies matched your strict filters.")
                else:
                    st.write("Vectorizing user profile & calculating cosine similarity...")
                    
                    # Construct Profile
                    industries = " ".join([("Hollywood" if l=='en' else l) for l in langs])
                    profile = f"{mood} {' '.join(genres)} {age} {industries}"
                    
                    rec_movie, score = get_recommendation(df_candidates, profile, st.session_state['seen_ids'])
                    
                    if rec_movie is None:
                        status.update(label="No unseen movies left!", state="warning")
                        st.warning("You've seen all the movies matching these filters!")
                    else:
                        status.update(label="Match Found!", state="complete")
                        st.session_state['current_rec'] = rec_movie
                        st.session_state['current_score'] = score
                        
                        # Log Preference
                        log_data = {
                            'user_id': st.session_state['user_id'],
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'genres': ', '.join(genres), 'languages': ', '.join(langs),
                            'age_filter': age, 'mood': mood,
                            'rec_id': rec_movie['id'], 'rec_title': rec_movie['title']
                        }
                        append_to_sheet('user_preferences', log_data)
                        append_to_sheet('recommendations_history', {'uid': st.session_state['user_id'], 'mid': rec_movie['id'], 'ts': log_data['timestamp']})

    # --- Display Recommendation ---
    if 'current_rec' in st.session_state:
        rec = st.session_state['current_rec']
        score = st.session_state['current_score']
        
        st.divider()
        col_img, col_info = st.columns([1, 2])
        
        with col_img:
            if rec['poster_path']:
                st.image(f"{IMAGE_BASE_URL}{rec['poster_path']}", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
        
        with col_info:
            st.markdown(f"## 🎬 {rec['title']}")
            st.caption(f"Released: {rec['release_date']} | Rating: ⭐ {rec['rating']}/10")
            
            # AI Confidence Bar
            st.write("### AI Match Confidence")
            st.progress(float(score), text=f"{int(score*100)}% Semantic Match")
            
            st.markdown("### Overview")
            st.info(rec['overview'])
            
            st.markdown("### Actions")
            c_yes, c_watch, c_skip = st.columns(3)
            
            if c_yes.button("✅ I've Seen It"):
                append_to_sheet('watched_history', {
                    'uid': st.session_state['user_id'], 'mid': rec['id'], 
                    'title': rec['title'], 'ts': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state['seen_ids'].add(rec['id'])
                del st.session_state['current_rec'] # Clear to force next search
                st.rerun()

            if c_watch.button("🍿 Watch Now"):
                append_to_sheet('watched_history', {
                    'uid': st.session_state['user_id'], 'mid': rec['id'], 
                    'title': rec['title'], 'ts': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state['seen_ids'].add(rec['id'])
                st.balloons()
                st.success(f"Added '{rec['title']}' to your history! Enjoy!")

            if c_skip.button("⏭️ Skip"):
                del st.session_state['current_rec']
                st.rerun()

if __name__ == '__main__':
    main()