import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import textwrap
import time
import streamlit.components.v1 as components
import plotly.graph_objects as go
import tempfile

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ADVANCED STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FloraMind | Computational Botany Interface",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Robust Entropy Calculation (Fallback if scipy is missing)
try:
    from scipy.stats import entropy
except ImportError:
    def entropy(pk, qk=None, base=None, axis=0):
        pk = np.asarray(pk)
        pk = 1.0 * pk / np.sum(pk, axis=axis, keepdims=True)
        vec = entropy(pk, qk, base, axis)
        return vec

# Custom CSS for "Vision Full" Aesthetic
st.markdown("""
<style>
    /* Global Theme & Animated Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(11, 15, 25) 90%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 400;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    /* Glassmorphism Card - Core Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px -1px rgba(99, 102, 241, 0.15);
    }
    
    /* Responsive Data Grid */
    .data-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    /* Metric Box Styling */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border-left: 2px solid #6366f1;
        transition: background 0.2s;
    }
    .metric-box:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .status-optimal { background: rgba(52, 211, 153, 0.1); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.2); }
    .status-warning { background: rgba(251, 191, 36, 0.1); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.2); }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 20px 0;
    }
    
    /* Remove default streamlit padding tweaks */
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        border: none;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. INTELLIGENT SYSTEM LOGIC & KNOWLEDGE BASE
# ---------------------------------------------------------------------

CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Enhanced Intellectual Knowledge Graph
BOTANICAL_DB = {
    'Daisy': {
        'sci_name': 'Bellis perennis',
        'family': 'Asteraceae',
        'order': 'Asterales',
        'traits': 'White ray florets, yellow disc florets',
        'symbolism': 'Innocence & Purity',
        'habitat': 'Lawns, meadows, disturbed ground',
        'description': "An archetypal species of the Asteraceae family. The name 'daisy' is considered a corruption of 'day's eye', because the whole head closes at night and opens in the morning."
    },
    'Dandelion': {
        'sci_name': 'Taraxacum officinale',
        'family': 'Asteraceae',
        'order': 'Asterales',
        'traits': 'Basal rosette leaves, hollow scape',
        'symbolism': 'Resilience & Healing',
        'habitat': 'Temperate zones, roadsides',
        'description': "A flowering herbaceous perennial known for its bright yellow flower heads that turn into round balls of silver-tufted fruits that disperse via wind."
    },
    'Rose': {
        'sci_name': 'Rosa rubiginosa',
        'family': 'Rosaceae',
        'order': 'Rosales',
        'traits': 'Sharp prickles, pinnate leaves',
        'symbolism': 'Love, Passion, Secrecy',
        'habitat': 'Gardens, hedgerows, thickets',
        'description': "A woody perennial of the genus Rosa. Roses are best known for their ornamental flowers and their prickles (often erroneously called thorns)."
    },
    'Sunflower': {
        'sci_name': 'Helianthus annuus',
        'family': 'Asteraceae',
        'order': 'Asterales',
        'traits': 'Heliotropism, massive flower head',
        'symbolism': 'Adoration & Loyalty',
        'habitat': 'Prairies, dry open areas',
        'description': "Famous for heliotropism, where young buds face the sun and track it across the sky. The 'flower' is actually a head of hundreds of tiny florets."
    },
    'Tulip': {
        'sci_name': 'Tulipa gesneriana',
        'family': 'Liliaceae',
        'order': 'Liliales',
        'traits': 'Cup-shaped tepals, bulbous geophyte',
        'symbolism': 'Perfect Love & Charity',
        'habitat': 'Steppes, mountain slopes',
        'description': "A spring-blooming perennial herbaceous bulbiferous geophyte. The flowers are usually large, showy, and actinomorphic (radially symmetric)."
    }
}

def find_file_in_dir(filename):
    """Robust file finder with path normalization"""
    if os.path.exists(filename): 
        return filename
    for root, dirs, files in os.walk('.'):
        if filename in files: 
            return os.path.join(root, filename)
    return None

@st.cache_resource
def load_inference_engine(uploaded_file=None):
    """
    Loads model securely. 
    Uses tempfile for uploads to avoid permission errors.
    """
    model = None
    
    # 1. Handle Uploaded File
    if uploaded_file is not None:
        try:
            # Create a temporary file to save the uploaded model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            model = tf.keras.models.load_model(tmp_path)
            # Cleanup temp file
            os.remove(tmp_path)
            return model
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded model: {e}")
            return None

    # 2. Handle Default File
    path = find_file_in_dir("flowers_mobilenetv2.keras")
    if path:
        try:
            model = tf.keras.models.load_model(path)
        except Exception as e:
            st.sidebar.error(f"Error loading system model: {e}")
            return None
            
    return model

def preprocess_specimen(image_data):
    """Scientific preprocessing pipeline for MobileNetV2"""
    # Resize with Lanczos for high quality
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(image).astype(np.float32)
    
    # MobileNetV2 expects [-1, 1] range
    # Formula: (x / 127.5) - 1.0
    normalized = (img_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

# ---------------------------------------------------------------------
# 3. SIDEBAR & HEADER
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ System Config")
    uploaded_model = st.file_uploader("Update Weights (.keras)", type="keras")
    
    with st.spinner("Initializing System..."):
        model = load_inference_engine(uploaded_model)
    
    st.markdown("---")
    st.markdown("**Session Log**")
    if 'history' not in st.session_state: st.session_state.history = []
    
    if st.session_state.history:
        for item in reversed(st.session_state.history[-3:]):
            st.caption(f"{item['time']} | {item['pred']}")
    else:
        st.caption("No recent analysis.")

st.markdown("# FloraMind <span style='font-size:0.4em; color:#818cf8; border:1px solid #818cf8; border-radius:4px; padding:2px 6px; vertical-align:middle; letter-spacing: 1px;'>LABS</span>", unsafe_allow_html=True)
st.markdown("##### Advanced Computational Botany & Taxonomy Interface")

# ---------------------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------------------

# Input Section
input_tab, sensor_tab = st.tabs(["📁 **Upload Specimen**", "📷 **Optical Sensor**"])
input_image = None

with input_tab:
    u_file = st.file_uploader("Upload high-res image for analysis", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if u_file: input_image = Image.open(u_file).convert("RGB")

with sensor_tab:
    c_file = st.camera_input("Capture live specimen")
    if c_file: input_image = Image.open(c_file).convert("RGB")

# ---------------------------------------------------------------------
# 5. ANALYSIS & VISUALIZATION
# ---------------------------------------------------------------------

if input_image and model:
    # --- LAYOUT: 2 Columns (Visual | Data) ---
    col_visual, col_data = st.columns([1.5, 2], gap="large")
    
    # 1. VISUAL COLUMN (Left)
    with col_visual:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image(input_image, use_container_width=True, caption="Specimen 00-A1 (Source Input)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Inference Logic
        processed = preprocess_specimen(input_image)
        start_t = time.time()
        preds = model.predict(processed)[0]
        inference_time = (time.time() - start_t) * 1000
        
        # Calculate Metrics
        top_idx = np.argmax(preds)
        top_prob = preds[top_idx]
        top_name = CLASS_NAMES[top_idx]
        uncertainty = entropy(preds)
        
        # --- NEW: VISUALIZATION STACK ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Tab 1: Radar (Tensor Shape), Tab 2: Bar (Distribution)
        v_tab1, v_tab2 = st.tabs(["🕸 Radar Scan", "📊 Distribution"])
        
        with v_tab1:
            # Interactive Radar Chart
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=preds, theta=CLASS_NAMES, fill='toself',
                line=dict(color='#818cf8', width=2),
                fillcolor='rgba(129, 140, 248, 0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=False, showline=False, gridcolor='rgba(255,255,255,0.1)'),
                    bgcolor='rgba(0,0,0,0)',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                showlegend=False,
                font=dict(color='#94a3b8', family="Inter")
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with v_tab2:
            # New Horizontal Bar Chart
            fig_bar = go.Figure(go.Bar(
                x=preds,
                y=CLASS_NAMES,
                orientation='h',
                marker=dict(color='rgba(129, 140, 248, 0.6)', line=dict(color='#818cf8', width=1)),
                text=[f"{p:.1%}" for p in preds],
                textposition='auto',
                hoverinfo='text+y'
            ))
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.1]),
                yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0')),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. DATA COLUMN (Right)
    with col_data:
        # Save to history
        if not st.session_state.history or st.session_state.history[-1]['pred'] != top_name:
            st.session_state.history.append({'time': time.strftime('%H:%M'), 'pred': top_name})

        # Logic for Status Badge
        badge_class = "status-optimal"
        badge_text = "CONFIRMED"
        if top_prob < 0.6:
            badge_class = "status-warning"
            badge_text = "UNCERTAIN"

        # --- TOP IDENTIFICATION CARD ---
        title_html = textwrap.dedent(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="metric-label">Primary Identification</span>
                    <h1 style="margin:0; font-size: 2.8rem; background: linear-gradient(90deg, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 20px rgba(167, 139, 250, 0.3));">
                        {top_name}
                    </h1>
                </div>
                <div class="{badge_class}">
                    {badge_text} {(top_prob*100):.1f}%
                </div>
            </div>
            
            <div class="divider"></div>
            
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 100px;">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value" style="color:#34d399;">{top_prob:.4f}</div>
                </div>
                <div style="flex: 1; min-width: 100px;">
                    <div class="metric-label">Entropy</div>
                    <div class="metric-value" style="color:#fbbf24;">{uncertainty:.3f}</div>
                </div>
                <div style="flex: 1; min-width: 100px;">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{inference_time:.0f} ms</div>
                </div>
            </div>
        </div>
        """)
        # Embed the identification card using a component iframe with inline CSS
        full_title_html = textwrap.dedent(f"""
        <style>
        .glass-card {{ background: rgba(30,41,59,0.4); border-radius:12px; padding:16px; color:#e2e8f0; font-family:Inter, sans-serif; }}
        .metric-label {{ font-size:0.75rem; text-transform:uppercase; color:#94a3b8; margin-bottom:6px; }}
        .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size:1.1rem; color:#f1f5f9; font-weight:600; }}
        .divider {{ height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin:12px 0; }}
        .status-optimal {{ background: rgba(52,211,153,0.08); color:#34d399; padding:6px 10px; border-radius:9999px; }}
        .status-warning {{ background: rgba(251,191,36,0.08); color:#fbbf24; padding:6px 10px; border-radius:9999px; }}
        </style>
        {title_html}
        """)
        components.html(full_title_html, height=220)

        # --- BOTANICAL KNOWLEDGE CARD ---
        info = BOTANICAL_DB.get(top_name, {})
        
        bio_html = textwrap.dedent(f"""
        <div class="glass-card">
            <h3 style="display:flex; align-items:center; gap:10px; margin-bottom: 20px;">
                📖 Botanical Dossier
            </h3>
            
            <div class="data-grid">
                <div class="metric-box">
                    <div class="metric-label">Scientific Name</div>
                    <div class="metric-value" style="font-style:italic;">{info.get('sci_name', 'N/A')}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-label">Taxonomy Family</div>
                    <div class="metric-value">{info.get('family', 'N/A')}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-label">Order</div>
                    <div class="metric-value">{info.get('order', 'N/A')}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-label">Symbolism</div>
                    <div class="metric-value" style="font-size: 0.9rem;">{info.get('symbolism', 'N/A')}</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 12px; border-left: 3px solid #38bdf8;">
                <div class="metric-label" style="margin-bottom: 8px;">Morphological Description</div>
                <p style="color: #cbd5e1; line-height: 1.6; font-size: 0.95rem; margin:0;">
                    {info.get('description', 'Data unavailable.')}
                </p>
            </div>
        </div>
        """)
        # Embed the botanical dossier card with inline CSS so styles appear inside the iframe
        full_bio_html = textwrap.dedent(f"""
        <style>
        .glass-card {{ background: rgba(30,41,59,0.4); border-radius:12px; padding:16px; color:#e2e8f0; font-family:Inter, sans-serif; }}
        .data-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); gap:12px; margin-top:8px; }}
        .metric-box {{ background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; border-left:2px solid #6366f1; }}
        .metric-label {{ font-size:0.75rem; text-transform:uppercase; color:#94a3b8; margin-bottom:6px; }}
        .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size:1.05rem; color:#f1f5f9; font-weight:600; }}
        .divider {{ height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin:12px 0; }}
        </style>
        {bio_html}
        """)
        components.html(full_bio_html, height=340)
        
        # Link Button using Class Name for search
        search_q = top_name.replace(' ', '+')
        st.link_button(
            label=f"🔎 Research '{top_name}' on Google", 
            url=f"https://google.com/search?q={search_q}",
            use_container_width=True
        )

elif not model:
    st.warning("⚠ Neural Core Offline. Please upload 'flowers_mobilenetv2.keras' via sidebar.")
else:
    # Idle State
    st.markdown("""
    <div style="text-align: center; margin-top: 100px; opacity: 0.5;">
        <h1>Waiting for Specimen</h1>
        <p>Upload an image or activate the optical sensor to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)