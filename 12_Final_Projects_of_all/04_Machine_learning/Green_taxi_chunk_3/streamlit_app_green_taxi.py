import streamlit as st
import warnings

# ==========================================
# 0. SYSTEM CONFIGURATION & WARNING SUPPRESSION
# ==========================================
# Suppress specific XGBoost/Pickle warnings to keep UI clean
warnings.filterwarnings("ignore", message=".*XGBoost.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import streamlit.components.v1 as components
import pydeck as pdk
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

st.set_page_config(
    page_title="NYC Smart Taxi Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 1. VISUAL STYLING (PREMIUM DARK MODE)
# ==========================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #262730;
        color: white;
        border: 1px solid #41444C;
        border-radius: 8px;
    }
    
    /* Custom Card Style */
    .glass-card {
        background: rgba(38, 39, 48, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Metric Text */
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #00CC96; /* Neon Green */
        margin-bottom: 0px;
        text-shadow: 0 0 10px rgba(0, 204, 150, 0.3);
    }
    .metric-label {
        font-size: 13px;
        color: #A3A8B8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(92.83deg, #00CC96 0%, #00b887 100%);
        color: white;
        border: none;
        height: 55px;
        font-size: 18px;
        font-weight: 800;
        border-radius: 12px;
        box-shadow: 0 10px 20px -10px rgba(0, 204, 150, 0.5);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 25px -10px rgba(0, 204, 150, 0.6);
    }
    
    /* Custom Badge for Coordinates */
    .coord-badge {
        background-color: #1F2229;
        color: #00CC96;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        border: 1px solid rgba(0, 204, 150, 0.3);
        display: inline-block;
        margin-top: 8px;
    }
    
    /* Headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #00CC96, #3793ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    h2, h3 {
        color: #FAFAFA !important;
    }
    
    /* Insights Box - Enhanced */
    .insight-box {
        background: linear-gradient(135deg, #262730 0%, #1F2229 100%);
        border-left: 5px solid #00CC96;
        padding: 20px;
        border-radius: 8px;
        color: #E6EAF1;
        font-size: 1rem;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .insight-header {
        color: #00CC96;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
        display: block;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC: GEOGRAPHY & MATH
# ==========================================

# Extended List of Locations (Massive Update)
NYC_LOCATIONS = {
    "Custom Coordinate": None,
    # Airports
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Newark Airport (EWR)": (40.6895, -74.1745),
    
    # Manhattan - Midtown/Uptown
    "Times Square": (40.7580, -73.9855),
    "Central Park (South)": (40.7681, -73.9718),
    "Empire State Building": (40.7484, -73.9857),
    "Rockefeller Center": (40.7587, -73.9787),
    "Bryant Park": (40.7536, -73.9832),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Penn Station": (40.7505, -73.9934),
    "Port Authority Bus Terminal": (40.7569, -73.9905),
    "MoMA (Museum of Modern Art)": (40.7614, -73.9776),
    "St. Patrick's Cathedral": (40.7584, -73.9759),
    "Carnegie Hall": (40.7651, -73.9799),
    "Lincoln Center": (40.7724, -73.9835),
    "American Museum of Natural History": (40.7813, -73.9740),
    "Guggenheim Museum": (40.7829, -73.9589),
    "Metropolitan Museum of Art": (40.7794, -73.9632),
    "Javits Center": (40.7580, -74.0021),
    "Columbia University": (40.8075, -73.9626),
    "Harlem (Apollo Theater)": (40.8101, -73.9501),
    
    # Manhattan - Downtown/Village
    "One World Trade Center": (40.7127, -74.0134),
    "Wall Street": (40.7074, -74.0113),
    "The Battery": (40.7033, -74.0170),
    "Staten Island Ferry Whitehall": (40.7014, -74.0132),
    "Flatiron Building": (40.7411, -73.9897),
    "Chelsea Market": (40.7420, -74.0048),
    "The High Line": (40.7480, -74.0048),
    "Whitney Museum": (40.7396, -74.0089),
    "Washington Square Park (NYU)": (40.7308, -73.9973),
    "SoHo": (40.7233, -74.0030),
    "Chinatown": (40.7158, -73.9974),
    
    # Brooklyn
    "Barclays Center": (40.6829, -73.9754),
    "Brooklyn Bridge Park": (40.7023, -73.9964),
    "Brooklyn Museum": (40.6712, -73.9636),
    "Brooklyn Botanic Garden": (40.6675, -73.9630),
    "Prospect Park": (40.6602, -73.9690),
    "Coney Island": (40.5744, -73.9785),
    "Williamsburg": (40.7128, -73.9619),
    
    # Queens
    "Citi Field": (40.7571, -73.8458),
    "Arthur Ashe Stadium": (40.7503, -73.8457),
    "Flushing Meadows Corona Park": (40.7400, -73.8407),
    "Queens Museum": (40.7458, -73.8467),
    "Astoria Park": (40.7797, -73.9224),
    
    # Bronx & Others
    "Yankee Stadium": (40.8296, -73.9262),
    "Bronx Zoo": (40.8506, -73.8770),
    "New York Botanical Garden": (40.8623, -73.8772),
    "Belmont Park (Nassau)": (40.7126, -73.7256),    
    "Yonkers (Westchester)": (40.9312, -73.8988)
}

# Updated Map
RATE_CODE_MAP = {
    1: "Standard Rate",
    2: "JFK Airport Rate",
    3: "Newark Airport Rate",
    4: "Nassau/Westchester Rate",
    5: "Negotiated Fare",
    6: "Group Ride Rate"
}

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula with City Traffic Multiplier (1.3x)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 
    return (c * r) * 1.3 

def get_smart_duration(distance, pickup_datetime, traffic_modifier=0):
    hour = pickup_datetime.hour
    is_weekend = pickup_datetime.dayofweek >= 5
    
    speed = 20
    traffic_status = "Normal Flow"
    
    if not is_weekend:
        if 7 <= hour < 10: 
            speed = 12
            traffic_status = "Heavy (Morning Rush)"
        elif 16 <= hour < 19: 
            speed = 10
            traffic_status = "Severe (Evening Rush)"
        elif 10 <= hour < 16: 
            speed = 15
            traffic_status = "Moderate Activity"
        elif hour >= 22 or hour < 5: 
            speed = 30
            traffic_status = "Clear Roads"
    else:
        if 12 <= hour < 20: 
            speed = 16
            traffic_status = "Weekend Traffic"

    speed += traffic_modifier
    if speed < 1: speed = 1
    
    duration_min = (distance / speed) * 60
    return duration_min, traffic_status

def get_time_surcharges(pickup_datetime, location_name=""):
    extra_charge = 0.0
    reason = []
    
    hour = pickup_datetime.hour
    day = pickup_datetime.dayofweek
    
    if 0 <= day <= 4 and 16 <= hour < 20:
        extra_charge += 1.00
        reason.append("Rush Hour (+$1.00)")
        
    if (hour >= 20) or (hour < 6):
        extra_charge += 0.50
        reason.append("Overnight (+$0.50)")
        
    if "Airport" in str(location_name):
        extra_charge += 1.25
        reason.append("Airport Fee (+$1.25)")
        
    return extra_charge, reason

def recommend_rate_code(pu_name, do_name):
    locations = str(pu_name) + str(do_name)
    if "JFK" in locations: return 2
    elif "Newark" in locations: return 3
    elif "Nassau" in locations or "Westchester" in locations: return 4
    else: return 1

def generate_smart_insight(dist, duration, rate_id, surcharges, arrival_time):
    """
    Generates a structured, visually attractive insight summary.
    """
    insight_html = f"""
    <div class="insight-header">💡 Smart Trip Insight</div>
    <ul style="margin: 0; padding-left: 20px; list-style-type: none; color: #E6EAF1;">
        <li style="margin-bottom: 8px;">📍 <b>Logistics:</b> {dist:.1f} miles • ~{duration:.0f} mins</li>
        <li style="margin-bottom: 8px;">🏁 <b>Arrival:</b> You should reach by <b>{arrival_time.strftime('%H:%M')}</b></li>
        <li style="margin-bottom: 8px;">💲 <b>Rate Applied:</b> {RATE_CODE_MAP[rate_id]}</li>
    """
    
    if surcharges:
        surcharge_text = ", ".join(surcharges)
        insight_html += f"<li>⚠️ <b>Surcharges:</b> <span style='color: #FF4B4B'>{surcharge_text}</span></li>"
    else:
        insight_html += f"<li>✅ <b>Surcharges:</b> None active</li>"
    
    insight_html += "</ul>"
    return insight_html

# ==========================================
# 3. LOAD RESOURCES (FIXED PATH FINDING)
# ==========================================

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
def load_model():
    filename = 'best_model_green_taxi_trained_on_100000_sample.pkl'
    model_path = find_file_in_dir(filename)
    
    if model_path:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {e}")
            return None
    else:
        st.error(f"❌ Model file '{filename}' not found. Please upload it to your repository.")
        return None

model = load_model()

def preprocess_input(data):
    dt = data['pickup_datetime']
    dist = data['trip_distance']
    dur = data['trip_duration_min']
    rc = data['RateCodeID']
    
    features = {
        'vendor_id': data['VendorID'],
        'Dropoff_latitude': data['dropoff_latitude'],
        'Passenger_count': data['passenger_count'],
        'Trip_distance': dist,
        'Extra': data['Extra'],
        'MTA_tax': data['MTA_tax'],
        'Tolls_amount': data['Tolls_amount'],
        'Pickup_longitude': data['pickup_longitude'],
        'Pickup_latitude': data['pickup_latitude'],
        'Dropoff_longitude': data['dropoff_longitude'],
        'trip_duration_min': dur,
        'pickup_hour': dt.hour,
        'pickup_day_of_week': dt.dayofweek,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
        'log_trip_distance': np.log1p(dist),
        'distance_duration_interaction': dist * dur,
        'hour_of_day': dt.hour,
        'Store_and_fwd_flag_Y': 0,
        'Payment_type_2': 1 if data['payment_type'] == 2 else 0,
        'Payment_type_3': 1 if data['payment_type'] == 3 else 0,
        'Payment_type_4': 1 if data['payment_type'] == 4 else 0,
        'Trip_type_2': 1 if data['trip_type'] == 2 else 0,
        'rate_code_2': 1 if rc == 2 else 0,
        'rate_code_3': 1 if rc == 3 else 0,
        'rate_code_4': 1 if rc == 4 else 0,
        'rate_code_5': 1 if rc == 5 else 0,
        'rate_code_6': 1 if rc == 6 else 0,
    }
    
    expected_order = [
        'vendor_id', 'Dropoff_latitude', 'Passenger_count', 'Trip_distance', 'Extra', 'MTA_tax', 
        'Tolls_amount', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'trip_duration_min', 
        'pickup_hour', 'pickup_day_of_week', 'is_weekend', 'log_trip_distance', 'distance_duration_interaction', 
        'hour_of_day', 'Store_and_fwd_flag_Y', 'Payment_type_2', 'Payment_type_3', 'Payment_type_4', 
        'Trip_type_2', 'rate_code_2', 'rate_code_3', 'rate_code_4', 'rate_code_5', 'rate_code_6'
    ]
    return pd.DataFrame([features])[expected_order]

# ==========================================
# 4. MAIN LAYOUT
# ==========================================

col_title, col_menu = st.columns([3, 1])

with col_title:
    st.title("🚖 NYC Smart Taxi Predictor")
    st.caption("v7.0 | Enhanced Locations | Smart Insight Engine")

with col_menu:
    page = st.selectbox("Navigation", ["Ride Predictor", "Data Analytics", "System Specs"], label_visibility="collapsed")

# ==========================================
# 5. PAGE: SMART PREDICTION
# ==========================================
if page == "Ride Predictor":
    
    # --- SESSION STATE INITIALIZATION ---
    if 'pu_lat' not in st.session_state: st.session_state.pu_lat = 40.6413
    if 'pu_lon' not in st.session_state: st.session_state.pu_lon = -73.7781
    if 'do_lat' not in st.session_state: st.session_state.do_lat = 40.7580
    if 'do_lon' not in st.session_state: st.session_state.do_lon = -73.9855

    # Callbacks
    def update_loc_from_preset(prefix):
        key_name, key_lat, key_lon = f"{prefix}_name", f"{prefix}_lat", f"{prefix}_lon"
        selection = st.session_state[key_name]
        if selection != "Custom Coordinate":
            st.session_state[key_lat], st.session_state[key_lon] = NYC_LOCATIONS[selection]

    def set_custom(prefix):
        st.session_state[f"{prefix}_name"] = "Custom Coordinate"

    # --- Container: Route Selection ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("1. Route & Time")
    
    col_loc1, col_loc2, col_time = st.columns([1.5, 1.5, 1])
    
    def render_loc_input(label, prefix, default_idx):
        st.selectbox(label, options=list(NYC_LOCATIONS.keys()), index=default_idx, key=f"{prefix}_name", on_change=update_loc_from_preset, args=(prefix,))
        if st.session_state[f"{prefix}_name"] == "Custom Coordinate":
            c1, c2 = st.columns(2)
            st.number_input("Lat", key=f"{prefix}_lat", format="%.4f", on_change=set_custom, args=(prefix,))
            st.number_input("Lon", key=f"{prefix}_lon", format="%.4f", on_change=set_custom, args=(prefix,))
        else:
            lat, lon = NYC_LOCATIONS[st.session_state[f"{prefix}_name"]]
            st.markdown(f'<div class="coord-badge">{lat:.4f}, {lon:.4f}</div>', unsafe_allow_html=True)
            
    with col_loc1: render_loc_input("Pickup Point", "pu", 1) # JFK Default
    with col_loc2: render_loc_input("Dropoff Point", "do", 3) # Times Sq Default

    with col_time:
        st.write("Pickup Schedule")
        d = st.date_input("Date", value=pd.to_datetime("2013-09-01"), label_visibility="collapsed")
        t = st.time_input("Time", value=pd.to_datetime("19:00").time(), label_visibility="collapsed")
        pickup_dt = pd.Timestamp.combine(d, t)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Calculations ---
    calc_distance = calculate_distance(st.session_state.pu_lat, st.session_state.pu_lon, st.session_state.do_lat, st.session_state.do_lon)
    smart_extra, surcharge_reason = get_time_surcharges(pickup_dt, st.session_state.pu_name)
    rec_rate_code = recommend_rate_code(st.session_state.pu_name, st.session_state.do_name)
    
    # --- Map Visual ---
    mid_lat = (st.session_state.pu_lat + st.session_state.do_lat) / 2
    mid_lon = (st.session_state.pu_lon + st.session_state.do_lon) / 2
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10.5, pitch=45)

    arc_data = pd.DataFrame([{"source": [st.session_state.pu_lon, st.session_state.pu_lat], "target": [st.session_state.do_lon, st.session_state.do_lat]}])
    scatter_data = pd.DataFrame([
        {"position": [st.session_state.pu_lon, st.session_state.pu_lat], "color": [0, 255, 150, 200], "radius": 300},
        {"position": [st.session_state.do_lon, st.session_state.do_lat], "color": [255, 75, 75, 200], "radius": 300}
    ])

    st.pydeck_chart(pdk.Deck(
        map_style=None, 
        initial_view_state=view_state,
        layers=[
            pdk.Layer("ArcLayer", data=arc_data, get_source="source", get_target="target", get_width=6, get_source_color=[0, 204, 150, 200], get_target_color=[255, 75, 75, 200]),
            pdk.Layer("ScatterplotLayer", data=scatter_data, get_position="position", get_color="color", get_radius="radius")
        ],
        tooltip={"html": "<b>Trip Path</b>"}
    ))

    # --- Container: Trip Details & Cost ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("2. Smart Estimation")
    
    col_config, col_result = st.columns([1, 1.2])
    
    with col_config:
        st.markdown("### ⚙️ Trip Config")
        passengers = st.slider("Passengers", 1, 6, 1)
        
        traffic_sim = st.select_slider(
            "Traffic Simulation (Impacts Duration)",
            options=[-5, -2, 0, 2, 5],
            value=0,
            format_func=lambda x: { -5: "Empty (-5mph)", -2: "Light (-2mph)", 0: "AI Predicted", 2: "Heavy (+2mph)", 5: "Gridlock (+5mph)" }[x]
        )
        
        est_duration_min, traffic_status = get_smart_duration(calc_distance, pickup_dt, traffic_sim)
        dropoff_dt = pickup_dt + timedelta(minutes=est_duration_min)
        
        rate_code = st.selectbox("Rate Code", options=list(RATE_CODE_MAP.keys()), index=list(RATE_CODE_MAP.keys()).index(rec_rate_code), format_func=lambda x: RATE_CODE_MAP[x])
        payment_type = st.selectbox("Payment", [1, 2], format_func=lambda x: "Credit Card" if x==1 else "Cash")
    
    with col_result:
        st.markdown("### 📊 Metrics")
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-value">{calc_distance:.1f}</div><div class="metric-label">Miles</div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-value">{est_duration_min:.0f}</div><div class="metric-label">Minutes</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;">
            <span style="color: #A3A8B8; font-size: 14px;">ESTIMATED DROP-OFF</span><br>
            <span style="color: #FAFAFA; font-size: 24px; font-weight: bold;">{dropoff_dt.strftime('%H:%M')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("CALCULATE FINAL FARE"):
            if model:
                try:
                    input_data = {
                        'VendorID': 2, 'RateCodeID': rate_code, 'passenger_count': passengers,
                        'trip_distance': calc_distance, 'pickup_longitude': st.session_state.pu_lon, 'pickup_latitude': st.session_state.pu_lat,
                        'dropoff_longitude': st.session_state.do_lon, 'dropoff_latitude': st.session_state.do_lat,
                        'payment_type': payment_type, 'pickup_datetime': pickup_dt,
                        'trip_type': 1, 'Extra': smart_extra, 'MTA_tax': 0.5, 'Tolls_amount': 0.0,
                        'trip_duration_min': est_duration_min
                    }
                    df = preprocess_input(input_data)
                    prediction = model.predict(df)[0]
                    st.session_state['last_pred'] = prediction
                    st.session_state['insight'] = generate_smart_insight(calc_distance, est_duration_min, rate_code, surcharge_reason, dropoff_dt)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model not loaded. Please check if the .pkl file is uploaded.")

        if 'last_pred' in st.session_state:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <h1 style="font-size: 60px; color: white; text-shadow: 0 0 20px rgba(0,204,150,0.5);">${st.session_state['last_pred']:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="insight-box">{st.session_state["insight"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Analytics":
    st.title("📊 Data Analytics")
    
    # Robust file finding for the report too
    report_filename = "Advanced_pandas_profiling_green_taxi_2013_chunk_3_report.html"
    report_path = find_file_in_dir(report_filename)
    
    if report_path:
        with open(report_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=1000, scrolling=True)
    else:
        st.warning(f"Analysis report file '{report_filename}' not found.")

elif page == "System Specs":
    st.title("🛠 System Specifications")
    st.markdown("""
    ### 🧠 AI Model
    - **Architecture:** XGBoost Regressor
    - **Training Data:** 100,000 Samples (2013 Green Taxi Dataset)
    - **Performance:** R² = 0.97
    
    ### ⚡ Smart Intelligence Features (v7.0)
    - **Extended Locations:** Over 40+ Landmarks, Museums, and Parks added.
    - **Smart Insight Engine:** Visually structured Markdown insights for better readability.
    - **Arrival Time:** Automatically calculates drop-off time.
    - **Traffic Simulation:** User-adjustable speed simulation.
    - **Contextual Pricing:** Advanced logic for detecting Surcharges and Rates.
    """)