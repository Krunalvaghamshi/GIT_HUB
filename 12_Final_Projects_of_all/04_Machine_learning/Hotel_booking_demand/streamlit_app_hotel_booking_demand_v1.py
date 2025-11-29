import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import warnings
import time
from datetime import date, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Hotel IQ | Revenue Concierge",
    page_icon="🛎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# --- 2. LUXURY HOTEL THEME (CSS) ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Playfair+Display:wght@400;600;700&display=swap');
    
    /* GLOBAL VARIABLES */
    :root {
        --primary: #1A253A; /* Midnight Blue */
        --accent: #C5A059;  /* Champagne Gold */
        --bg: #F9F8F6;      /* Alabaster */
        --text: #2C3E50;    /* Charcoal */
        --success: #166534;
        --danger: #991B1B;
    }

    /* RESET & BACKGROUND */
    .stApp {
        background-color: var(--bg);
        font-family: 'Lato', sans-serif;
        color: var(--text);
    }
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif;
        color: var(--primary);
        letter-spacing: 0.5px;
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: var(--primary);
        border-right: 4px solid var(--accent);
    }
    
    /* SIDEBAR TEXT OVERRIDES */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-family: 'Playfair Display', serif;
    }
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div {
        color: #E2E8F0 !important; /* Light text for dark sidebar */
    }
    
    /* WIDGET STYLING */
    div[data-baseweb="select"] > div {
        border-radius: 4px;
        border-color: #CBD5E1;
    }
    
    /* LUXURY CARDS */
    .hotel-card {
        background: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.08);
        border-top: 4px solid var(--accent);
        margin-bottom: 24px;
    }
    
    /* BUTTONS: GOLD GRADIENT */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #C5A059 100%);
        color: white !important;
        border: none;
        padding: 14px 28px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
        border-radius: 4px;
        box-shadow: 0 4px 15px rgba(197, 160, 89, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(197, 160, 89, 0.5);
    }
    
    /* BADGES & TAGS */
    .status-badge {
        font-family: 'Lato', sans-serif;
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 50px;
        letter-spacing: 0.1em;
    }
    
    /* KPI METRICS */
    .kpi-label {
        font-family: 'Lato', sans-serif;
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }
    .kpi-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: var(--primary);
        font-weight: 700;
    }
    
    /* SMART RECOMMENDATION LIST */
    .rec-card {
        border-left: 3px solid var(--primary);
        background: #F8FAFC;
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
    .rec-title {
        font-weight: 700;
        color: var(--primary);
        font-size: 0.95rem;
        margin-bottom: 4px;
        display: block;
    }
    .rec-body {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    /* DRIVER BARS */
    .driver-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.85rem;
        border-bottom: 1px dashed #E2E8F0;
        padding-bottom: 4px;
    }
    .driver-pos { color: var(--success); font-weight: 600; }
    .driver-neg { color: var(--danger); font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# --- 3. LOGIC & MAPPINGS ---
HOTEL_MAP = {"Resort Hotel": 0, "City Hotel": 1}

MEAL_MAP = {
    "Bed & Breakfast (BB)": 0, "Full Board (FB)": 1, 
    "Half Board (HB)": 2, "Self Catering": 3, "Undefined": 4
}
MARKET_MAP = {
    "Aviation": 0, "Complementary": 1, "Corporate": 2, "Direct": 3, 
    "Group": 4, "Offline Travel Agent": 5, "Online Travel Agent (OTA)": 6
}
DIST_MAP = {
    "Corporate": 0, "Direct": 1, "GDS": 2, "Travel Agents": 3
}
DEPOSIT_MAP = {
    "No Deposit": 0, "Non-Refundable": 1, "Refundable": 2
}

# --- 4. LOAD RESOURCES (FIXED PATH FINDING) ---

# Robust function to find files in cloud environment
def find_file_in_dir(filename):
    if os.path.exists(filename):
        return filename
    # Search current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    return None

@st.cache_resource
def load_system(uploaded_file=None):
    """
    Tries to load model from:
    1. Uploaded file (Manual Sync)
    2. Robust Path Finding (Automatic Sync)
    """
    if uploaded_file is not None:
        try:
            return joblib.load(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    
    filename = "xgboost_booking_model.pkl"
    model_path = find_file_in_dir(filename)
    
    if model_path:
        try:
            return joblib.load(model_path)
        except Exception as e:
            # Silent fail to allow fallback to manual upload if needed
            return None
    return None

def calculate_stay_details(check_in, check_out):
    total = (check_out - check_in).days
    we, wk = 0, 0
    curr = check_in
    while curr < check_out:
        if curr.weekday() >= 5: we += 1
        else: wk += 1
        curr += timedelta(days=1)
    return total, wk, we

def preprocess_and_predict(model, scaler, raw_inputs, feature_names):
    df = pd.DataFrame([raw_inputs], columns=feature_names)
    # Smart Log Transform
    if 'lead_time' in df.columns: df['lead_time'] = np.log1p(df['lead_time'])
    if 'adr' in df.columns: df['adr'] = np.log1p(df['adr'])
    scaled = scaler.transform(df)
    return model.predict_proba(scaled)[0][1]

def analyze_risk_factors(inputs, prob):
    """
    Intelligent Driver Analysis: Break down WHY the risk is high/low.
    Returns: Two lists (Amplifiers, Stabilizers)
    """
    amplifiers = []
    stabilizers = []
    
    # 1. Lead Time Analysis
    if inputs['lead_time'] > 150:
        amplifiers.append(f"Long Lead Time ({inputs['lead_time']} days) increases uncertainty")
    elif inputs['lead_time'] < 7:
        stabilizers.append("Imminent Arrival (Last minute booking)")
        
    # 2. Market Segment
    m_seg_code = inputs['market_segment']
    # Code 6 = Online TA (High Risk), Code 2 = Corporate (Stable)
    if m_seg_code == 6:
        amplifiers.append("Online TA bookings have higher cancel rates")
    elif m_seg_code == 2:
        stabilizers.append("Corporate bookings are typically stable")
    elif m_seg_code == 3:
        stabilizers.append("Direct bookings show high intent")
        
    # 3. Special Requests (Strong signal of intent)
    if inputs['total_of_special_requests'] > 0:
        stabilizers.append(f"Guest made {inputs['total_of_special_requests']} special request(s) (Shows intent)")
    else:
        if prob > 0.6: amplifiers.append("No special requests made")
        
    # 4. History
    if inputs['is_repeated_guest'] == 1:
        stabilizers.append("Repeat Guest (Loyalty factor)")
        
    # 5. Parking
    if inputs['required_car_parking_spaces'] > 0:
        stabilizers.append("Parking required (Implies logistics are planned)")
        
    # 6. Seasonality (Rough heuristic for Portugal data)
    # Summer (July/Aug) is high demand, Winter is lower
    month = inputs['arrival_date_month']
    if month in [7, 8]:
        stabilizers.append("Peak Season (High Demand)")
        
    return amplifiers, stabilizers

def get_concierge_advice(inputs, prob):
    """
    Generates Context-Aware Recommendations.
    """
    recs = []
    
    # --- CONTEXT 1: HIGH VALUE GUEST ---
    if inputs['adr'] > 250:
        recs.append({
            "title": "💎 VIP Handling Protocol",
            "body": "ADR exceeds standard threshold. Assign 'Guest Relations Manager' to welcome. Pre-allocate room with preferred view to ensure value perception."
        })
        
    # --- CONTEXT 2: FAMILY VS BUSINESS ---
    has_kids = inputs['children'] > 0 or inputs['babies'] > 0
    is_corporate = inputs['market_segment'] == 2 # Corporate
    
    if has_kids:
        recs.append({
            "title": "🧸 Family Logistics",
            "body": "Ensure housekeeping prepares extra bedding/crib prior to arrival. Suggest 'Family Activity Guide' in welcome email."
        })
    elif is_corporate:
        recs.append({
            "title": "💼 Business Express",
            "body": "Guest likely values speed. Pre-print invoice and prepare Express Check-in key packet."
        })
        
    # --- CONTEXT 3: RISK MITIGATION ---
    if prob > 0.7:
        recs.append({
            "title": "🛡️ Critical Retention Action",
            "body": "High cancellation probability. 1) Verify credit card validity immediately. 2) Attempt to convert to Non-Refundable with a 5% discount offer."
        })
    elif prob > 0.4:
        recs.append({
            "title": "📧 Engagement Nurturing",
            "body": "Moderate risk. Send a personalized 'Pre-Arrival Concierge' email offering dinner reservations or spa booking to lock in commitment."
        })
        
    # --- CONTEXT 4: LEAD TIME SPECIFIC ---
    if inputs['lead_time'] > 120:
        recs.append({
            "title": "📅 Long-Lead Reconfirmation",
            "body": "Booking made over 4 months ago. Plans change. Send a subtle 'We are looking forward to seeing you' message to gauge responsiveness."
        })
        
    # Default
    if not recs:
        recs.append({
            "title": "✅ Standard Procedure",
            "body": "Booking profile is healthy. Proceed with standard operational preparation."
        })
        
    return recs

# --- 5. SIDEBAR (THE "FRONT DESK") ---

def sidebar_front_desk():
    st.sidebar.markdown("<div style='margin-bottom: 20px;'><h1 style='font-size:24px; color:white;'>🛎️ Front Desk</h1></div>", unsafe_allow_html=True)
    
    inputs = {}
    
    with st.sidebar.expander("DATE & PROPERTY", expanded=True):
        inputs['hotel'] = HOTEL_MAP[st.selectbox("Property", list(HOTEL_MAP.keys()), index=1)]
        
        today = date.today()
        c_in = st.date_input("Check-In", today + timedelta(days=30))
        c_out = st.date_input("Check-Out", today + timedelta(days=34))
        b_date = st.date_input("Booking Date", today)
        
        # Logic
        lead = (c_in - b_date).days
        tot, wk, we = calculate_stay_details(c_in, c_out)
        
        inputs['lead_time'] = lead
        inputs['arrival_date_month'] = c_in.month
        inputs['arrival_date_week_number'] = c_in.isocalendar()[1]
        inputs['arrival_date_day_of_month'] = c_in.day
        inputs['stays_in_week_nights'] = wk
        inputs['stays_in_weekend_nights'] = we
        
        # Hidden
        inputs['month'] = c_in.month
        inputs['day'] = c_in.day
        inputs['year'] = c_in.year

    with st.sidebar.expander("GUEST DETAILS"):
        c1, c2 = st.columns(2)
        inputs['adults'] = c1.number_input("Adults", 1, 10, 2)
        inputs['children'] = c2.number_input("Kids", 0, 10, 0)
        inputs['babies'] = 0
        
    with st.sidebar.expander("REVENUE & SEGMENT", expanded=True):
        inputs['meal'] = MEAL_MAP[st.selectbox("Meal Plan", list(MEAL_MAP.keys()))]
        inputs['market_segment'] = MARKET_MAP[st.selectbox("Segment", list(MARKET_MAP.keys()), index=6)]
        inputs['distribution_channel'] = DIST_MAP[st.selectbox("Channel", list(DIST_MAP.keys()), index=1)]
        inputs['adr'] = st.number_input("Rate (ADR $)", 0, 5000, 150)
        inputs['deposit_type'] = DEPOSIT_MAP[st.selectbox("Deposit", list(DEPOSIT_MAP.keys()))]
        
    with st.sidebar.expander("HISTORY & REQUESTS"):
        inputs['is_repeated_guest'] = 1 if st.checkbox("Repeat Guest") else 0
        inputs['previous_cancellations'] = st.number_input("Prior Cancels", 0, 10, 0)
        inputs['total_of_special_requests'] = st.slider("Requests", 0, 5, 1)
        inputs['required_car_parking_spaces'] = st.slider("Parking", 0, 2, 0)

    # Defaults
    defaults = {
        'booking_changes': 0, 'customer_type': 2, 'previous_bookings_not_canceled': 0, 
        'reserved_room_type': 0, 'assigned_room_type': 0, 'agent': 0, 'company': 0, 
        'days_in_waiting_list': 0
    }
    inputs.update(defaults)
    
    return inputs

# --- 6. MAIN FLOOR ---

def main():
    # HEADER
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<h1 style='font-size: 42px; margin-bottom: 0;'>Hotel IQ</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: Playfair Display; font-style: italic; font-size: 18px; color: #64748b;'>Revenue & Risk Concierge</p>", unsafe_allow_html=True)
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933942.png", width=60) # Elegant Bell Icon

    # --- FAIL-SAFE LOADING SYSTEM ---
    # 1. Check for manual upload first
    uploaded_file = st.sidebar.file_uploader("Admin: Upload Model (.pkl)", type="pkl")
    
    # 2. Try loading with robust path finding
    artifacts = load_system(uploaded_file)
    
    if not artifacts:
        st.error("⚠️ SYSTEM OFFLINE: Model file not found.")
        st.info("Please ensure 'xgboost_booking_model.pkl' is in your GitHub repo, OR upload it manually in the sidebar to sync.")
        st.stop()
        
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts.get('feature_names', getattr(model, 'feature_names_in_', []))

    # SIDEBAR
    inputs = sidebar_front_desk()
    
    st.markdown("<hr style='border-top: 1px solid #E2E8F0; margin: 30px 0;'>", unsafe_allow_html=True)

    # ACTION BUTTON
    if 'pred' not in st.session_state: st.session_state.pred = None
    
    if st.button("Analyze Reservation Risk"):
        with st.spinner("Consulting Revenue Engine..."):
            time.sleep(0.5)
            ord_vals = [inputs.get(f, 0) for f in feature_names]
            prob = preprocess_and_predict(model, scaler, ord_vals, feature_names)
            st.session_state.pred = prob

    # RESULTS
    if st.session_state.pred is not None:
        prob = st.session_state.pred
        amplifiers, stabilizers = analyze_risk_factors(inputs, prob)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- CARDS LAYOUT ---
        col_risk, col_detail = st.columns([1, 1.4], gap="large")
        
        # COLUMN 1: RISK GAUGE & DRIVERS
        with col_risk:
            st.markdown("<div class='hotel-card'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; margin-bottom: 20px;'><span class='kpi-label'>CANCELLATION PROBABILITY</span></div>", unsafe_allow_html=True)
            
            # ELEGANT GAUGE
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 48, 'family': "Playfair Display", 'color': "#1A253A"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 0},
                    'bar': {'color': "rgba(0,0,0,0)"}, # Transparent
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 100], 'color': "#F1F5F9"}, # Track
                        {'range': [0, prob*100], 'color': "#1A253A" if prob < 0.4 else "#C5A059" if prob < 0.7 else "#8B0000"} # Navy/Gold/DeepRed
                    ],
                    'threshold': {
                        'line': {'color': "#1A253A", 'width': 2},
                        'thickness': 1.0, 'value': prob * 100
                    }
                }
            ))
            fig.update_layout(height=180, margin=dict(l=20,r=20,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4>Risk Factor Analysis</h4>", unsafe_allow_html=True)
            
            if stabilizers:
                st.markdown("<span class='kpi-label' style='color:#166534;'>🟢 STABILITY SIGNALS</span>", unsafe_allow_html=True)
                for item in stabilizers:
                    st.markdown(f"<div class='driver-row'><span style='color:#334155'>{item}</span></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
            if amplifiers:
                st.markdown("<span class='kpi-label' style='color:#991B1B;'>🔴 RISK AMPLIFIERS</span>", unsafe_allow_html=True)
                for item in amplifiers:
                    st.markdown(f"<div class='driver-row'><span style='color:#334155'>{item}</span></div>", unsafe_allow_html=True)
            
            if not stabilizers and not amplifiers:
                 st.markdown("<div class='driver-row'>No significant anomalies detected.</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # COLUMN 2: SMART RECOMMENDATIONS
        with col_detail:
            st.markdown("<div class='hotel-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Concierge Action Plan</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.9rem; color:#64748b;'>AI-Generated personalized protocols for this reservation.</p>", unsafe_allow_html=True)
            
            recs = get_concierge_advice(inputs, prob)
            for r in recs:
                st.markdown(f"""
                <div class='rec-card'>
                    <span class='rec-title'>{r['title']}</span>
                    <span class='rec-body'>{r['body']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # SIMULATOR (Strategy Lab)
        st.markdown("<h3>🧪 Strategy Lab</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='hotel-card' style='border-top-color: #1A253A;'>", unsafe_allow_html=True)
            
            s1, s2, s3 = st.columns([1.5, 1.5, 1])
            with s1:
                st.markdown("<span class='kpi-label'>ADJUST RATE (ADR)</span>", unsafe_allow_html=True)
                sim_adr = st.slider("", 50, 500, int(inputs['adr']), label_visibility="collapsed")
            with s2:
                st.markdown("<span class='kpi-label'>ADJUST LEAD TIME</span>", unsafe_allow_html=True)
                sim_lead = st.slider("", 0, 365, int(inputs['lead_time']), label_visibility="collapsed")
            
            # Sim Logic
            sim_in = inputs.copy()
            sim_in['adr'] = sim_adr
            sim_in['lead_time'] = sim_lead
            sim_ord = [sim_in.get(f, 0) for f in feature_names]
            sim_prob = preprocess_and_predict(model, scaler, sim_ord, feature_names)
            diff = (sim_prob - prob) * 100
            
            with s3:
                st.markdown(f"""
                <div style='text-align: right;'>
                    <div class='kpi-label'>NEW RISK</div>
                    <div class='kpi-value'>{sim_prob*100:.1f}%</div>
                    <div style='font-size: 14px; font-weight: 700; color: {'#166534' if diff < 0 else '#991B1B'};'>{diff:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()