# 🎬 Project Title: CineMind AI | Semantic Movie Guide

### **Project Overview**
**CineMind AI** is an advanced, end-to-end movie recommendation engine that moves beyond simple genre filtering to understand the *semantic meaning* of a user's request. Unlike traditional systems that match tags ("Action" to "Action"), CineMind AI utilizes Natural Language Processing (NLP) and Neural Embeddings to interpret complex user moods (e.g., *"I want something inspiring but tragic"*) and match them with movies that share that specific emotional and thematic DNA.

The project is a full-stack data science application transformed from a local script into a cloud-native, persistent web application. It solves the critical "stateless" challenge of free cloud hosting by utilizing a custom-engineered NoSQL database layer built on top of the Google Sheets API.

---

### **✨ Key Features & Capabilities**

1.  **Semantic Search Engine:**
    * Uses **Hugging Face Transformers** (`sentence-transformers/all-MiniLM-L6-v2`) to convert user inputs (mood, genre preferences, era) into high-dimensional vector embeddings.
    * Calculates **Cosine Similarity** between the user's "Mood Vector" and movie descriptions to find the closest semantic match, not just keyword matches.

2.  **Smart Discovery System:**
    * Integrates with the **TMDB (The Movie Database) API** to fetch real-time movie data, posters, ratings, and overviews.
    * Filters by Genre, Language (supporting regional Indian industries like Telugu/Tamil/Hindi alongside Hollywood), and Release Era (Last 5, 10, or 25 years).

3.  **"Vision-Full" UI/UX:**
    * Features a custom-styled Streamlit interface with a "Dark/Glassmorphism" aesthetic (`linear-gradient` backgrounds, translucent cards).
    * Includes an **AI Confidence Bar**, showing the user exactly how confident the model is in the recommendation (e.g., "85% Semantic Match").

4.  **Cloud Persistence Architecture (The "Secret Sauce"):**
    * **Problem:** Free cloud hosting (like Streamlit Community Cloud) is ephemeral; it wipes local CSV files on reboot.
    * **Solution:** The app connects to **Google Sheets** via a GCP Service Account. It uses the sheet as a real-time, persistent database to store User Profiles, Watch History, and Preferences, ensuring data survives app restarts.

---

### **🛠️ Technical Architecture**

* **Frontend:** Streamlit (Python) with custom CSS/HTML injection.
* **Backend Logic:** Python 3.11.
* **Machine Learning:**
    * **Library:** PyTorch, Transformers.
    * **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Optimized for sentence embeddings).
* **Data Source:** TMDB API (The Movie Database).
* **Database (Persistence):** Google Sheets API (via `gspread` and `oauth2client`).
* **Security:** Streamlit Secrets management (`secrets.toml`) to handle API keys and GCP credentials securely.

---

### **🚀 The Development Journey**

The development of CineMind AI followed a rigorous 5-phase engineering lifecycle, evolving from a simple script to a robust cloud application.

#### **Phase 1: The Foundation (CLI Prototype)**
* **Goal:** Prove the concept of semantic search.
* **Action:** I built a command-line interface (CLI) script using `pandas` and `torch`.
* **Logic:** The system took user inputs via `input()`, generated embeddings, and printed results to the console. Data was stored in local CSV files like `users.csv`.

#### **Phase 2: Interface Transformation**
* **Goal:** Create a user-friendly GUI.
* **Action:** Migrated the logic to **Streamlit**.
* **Innovation:** Replaced `while` loops with Streamlit's `Session State` to manage user flow (Login -> Filter -> Recommend -> Rate) without refreshing the entire page variables.

#### **Phase 3: The "Amnesia" Crisis**
* **Challenge:** Upon deploying to the cloud, I discovered the "Ephemeral File System" issue. Every time the app went to sleep or rebooted, the `users.csv` and `watched_history.csv` files were reset, causing users to lose their accounts and history.

#### **Phase 4: Cloud Persistence Architecture**
* **Solution:** I re-engineered the backend to decouple storage from the application container.
* **Implementation:**
    * Created a **Google Cloud Platform (GCP)** Service Account ("Robot User").
    * Implemented `gspread` to read/write to a Google Sheet named `Movie_Recommender_DB`.
    * Used **Streamlit Secrets** to inject the JSON credentials at runtime, keeping the repository secure.

#### **Phase 5: Intelligence & UX Refinement**
* **Goal:** Polish the experience (v2.0 Stable).
* **Refinements:**
    * **Loop Prevention:** Implemented logic to track `seen_ids` in real-time so the AI never recommends a movie the user has already watched or skipped.
    * **Visuals:** Added high-res movie posters and a "Mood" input field to fully leverage the NLP model.

---

### **📊 Database Schema & Logging**

The system maintains detailed logs to improve future recommendations. Based on the project files, the data structure is:

1.  **Users Table:** `user_id`, `username` (Handles authentication).
2.  **Watched History:**
    * Tracks what users have actually seen.
    * *Columns:* `user_id`, `movie_id`, `title`, `timestamp`.
3.  **User Preferences (The "Context" Log):**
    * This dataset is valuable for future model fine-tuning. It records *what* the user asked for versus *what* they got.
    * *Columns:* `user_id`, `genres`, `languages`, `age_filter`, `mood`, `recommended_movie_title`.
4.  **Recommendation History:**
    * Tracks every suggestion made by the AI to prevent duplicates.

---

### **🔗 Project Links**

**📂 GitHub Repository:**
https://github.com/Krunalvaghamshi/GIT_HUB/tree/7e9c1307bad179b2d433ee70e1c298f56726febd/12_Final_Projects_of_all/05_NLP/movie_recommendation_system

**🌐 Live Application:**
https://app-n9reblxcyanyfqshfdykcm.streamlit.app/

**📄 Project Documentation:**
https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/documentation_of_movie_recommendation.html

**💼 Portfolio:**
https://kruvs-portfolio.vercel.app/

