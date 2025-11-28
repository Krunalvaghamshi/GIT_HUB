import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import warnings

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Hotel IQ | Prediction Console",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# --- 2. CSS STYLING (High Contrast & Professional) ---
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #f8fafc; /* Very light slate background */
    }
    
    /* Text Hierarchy */
    h1 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    h2, h3 { color: #334155; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    p, label { color: #475569; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #0f172a !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    section[data-testid="stSidebar"] div.stMarkdown h1, 
    section[data-testid="stSidebar"] div.stMarkdown h2, 
    section[data-testid="stSidebar"] div.stMarkdown h3,
    section[data-testid="stSidebar"] div.stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important; /* White text for sidebar */
    }
    
    /* Risk Alert Boxes - High Contrast */
    .risk-box-critical {
        background-color: #fee2e2; /* Light Red */
        border-left: 5px solid #dc2626; /* Dark Red */
        padding: 20px;
        border-radius: 8px;
        color: #991b1b; /* Dark Red Text */
        margin-bottom: 20px;
    }
    .risk-box-moderate {
        background-color: #ffedd5; /* Light Orange */
        border-left: 5px solid #ea580c; /* Dark Orange */
        padding: 20px;
        border-radius: 8px;
        color: #9a3412; /* Dark Orange Text */
        margin-bottom: 20px;
    }
    .risk-box-safe {
        background-color: #dcfce7; /* Light Green */
        border-left: 5px solid #16a34a; /* Dark Green */
        padding: 20px;
        border-radius: 8px;
        color: #166534; /* Dark Green Text */
        margin-bottom: 20px;
    }
    
    /* Simulation Area */
    .sim-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTS ---
HOTEL_MAPPING = {"Resort Hotel": 0, "City Hotel": 1}
MONTH_MAPPING = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
MEAL_MAPPING = {"BB": 0, "FB": 1, "HB": 2, "SC": 3, "Undefined": 4}
MARKET_SEGMENT_MAPPING = {"Aviation": 0, "Complementary": 1, "Corporate": 2, "Direct": 3, "Groups": 4, "Offline TA/TO": 5, "Online TA": 6}
DISTRIBUTION_MAPPING = {"Corporate": 0, "Direct": 1, "GDS": 2, "TA/TO": 3}
DEPOSIT_MAPPING = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
CUSTOMER_TYPE_MAPPING = {"Contract": 0, "Group": 1, "Transient": 2, "Transient-Party": 3}

# --- 4. BACKEND ---

@st.cache_data
def load_data():
    # Only strictly necessary data loading
    if os.path.exists("hotel_bookings.csv"):
        return pd.read_csv("hotel_bookings.csv") # Kept for potential validation
    return None

@st.cache_resource
def load_system():
    filename = "xgboost_booking_model.pkl"
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except: return None
    return None

def predict_risk(model, scaler, inputs, feature_names):
    """Encapsulates prediction logic for reuse in simulator."""
    # Ensure inputs match feature order
    input_vector = pd.DataFrame([inputs], columns=feature_names)
    scaled_vector = scaler.transform(input_vector)
    prob = model.predict_proba(scaled_vector)[0][1]
    return prob

# --- 5. UI COMPONENTS ---

def sidebar_inputs():
    st.sidebar.title("🛠️ Booking Parameters")
    
    inputs = {}
    
    # 1. Hotel & Time
    st.sidebar.subheader("1. Property & Time")
    hotel = st.sidebar.selectbox("Hotel Type", list(HOTEL_MAPPING.keys()), index=1) # Default City Hotel
    inputs['hotel'] = HOTEL_MAPPING[hotel]
    
    inputs['lead_time'] = st.sidebar.slider("Lead Time (Days)", 0, 365, 45)
    
    month = st.sidebar.selectbox("Arrival Month", list(MONTH_MAPPING.keys()), index=1) # Default February
    inputs['arrival_date_month'] = MONTH_MAPPING[month]
    inputs['month'] = inputs['arrival_date_month']
    
    # Hidden defaults for date
    inputs['arrival_date_week_number'] = 30 
    inputs['arrival_date_day_of_month'] = 15
    inputs['day'] = 15
    inputs['year'] = 2017

    # 2. Guest Profile
    st.sidebar.subheader("2. Guest Profile")
    inputs['adults'] = st.sidebar.number_input("Adults", 1, 10, 2)
    inputs['children'] = st.sidebar.number_input("Children", 0, 10, 0)
    inputs['babies'] = 0
    
    c1, c2 = st.sidebar.columns(2)
    with c1: inputs['stays_in_week_nights'] = st.number_input("Week Nights", 0, 10, 2)
    with c2: inputs['stays_in_weekend_nights'] = st.number_input("Wknd Nights", 0, 10, 0)
    
    meal = st.sidebar.selectbox("Meal Plan", list(MEAL_MAPPING.keys()), index=1) # Default FB based on prompt
    inputs['meal'] = MEAL_MAPPING[meal]

    # 3. Financials
    st.sidebar.subheader("3. Financials & Channel")
    market = st.sidebar.selectbox("Market Segment", list(MARKET_SEGMENT_MAPPING.keys()), index=5) # Offline TA/TO
    inputs['market_segment'] = MARKET_SEGMENT_MAPPING[market]
    
    dist = st.sidebar.selectbox("Channel", list(DISTRIBUTION_MAPPING.keys()), index=1) # Direct
    inputs['distribution_channel'] = DISTRIBUTION_MAPPING[dist]
    
    inputs['adr'] = st.sidebar.number_input("ADR ($ Price)", 50, 1000, 100)
    
    dep = st.sidebar.selectbox("Deposit Type", list(DEPOSIT_MAPPING.keys()), index=1) # Non Refund
    inputs['deposit_type'] = DEPOSIT_MAPPING[dep]

    # 4. History
    st.sidebar.subheader("4. History & Requests")
    inputs['is_repeated_guest'] = 0
    inputs['previous_cancellations'] = st.sidebar.number_input("Prev. Cancels", 0, 20, 0)
    inputs['total_of_special_requests'] = st.sidebar.slider("Special Requests", 0, 5, 0)
    inputs['required_car_parking_spaces'] = st.sidebar.slider("Parking Spaces", 0, 3, 0)
    
    # Defaults
    inputs['booking_changes'] = 0
    inputs['customer_type'] = CUSTOMER_TYPE_MAPPING["Transient"]
    inputs['previous_bookings_not_canceled'] = 0
    inputs['reserved_room_type'] = 0 
    inputs['assigned_room_type'] = 0
    inputs['agent'] = 0
    inputs['company'] = 0
    inputs['days_in_waiting_list'] = 0

    return inputs

# --- 6. MAIN APP LOGIC ---

def main():
    # --- HEADER ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Hotel IQ: Enterprise Risk Engine")
        st.caption("Predictive Revenue Management & Cancellation Forecaster")
    with col_h2:
        st.success("🟢 System Online")

    # --- LOAD SYSTEM ---
    artifacts = load_system()
    if not artifacts:
        st.error("⚠️ Model file not found. Please upload 'xgboost_booking_model.pkl'.")
        st.stop()
        
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts.get('feature_names', getattr(model, 'feature_names_in_', []))

    # --- INPUTS ---
    inputs = sidebar_inputs()
    
    # --- PREDICTION ---
    # Construct ordered input list
    ordered_inputs = {k: inputs.get(k, 0) for k in feature_names}
    input_values = [ordered_inputs[k] for k in feature_names]
    
    current_prob = predict_risk(model, scaler, input_values, feature_names)
    
    # --- CALCULATE FINANCIALS ---
    nights = inputs['stays_in_week_nights'] + inputs['stays_in_weekend_nights']
    gross_val = nights * inputs['adr']
    exp_loss = gross_val * current_prob
    net_val = gross_val - exp_loss

    # --- MAIN DISPLAY AREA ---
    
    # 1. VISUAL GAUGE (The "Vision Full" Part)
    col_viz, col_metrics = st.columns([1.2, 1])
    
    with col_viz:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cancellation Probability"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"}, # Hide default bar, use threshold
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': "#dcfce7"},
                    {'range': [40, 70], 'color': "#ffedd5"},
                    {'range': [70, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "#1e293b", 'width': 4},
                    'thickness': 0.75,
                    'value': current_prob * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.markdown("### Financial Impact")
        st.metric("Gross Booking Value", f"${gross_val:,.2f}")
        st.metric("Expected Loss", f"-${exp_loss:,.2f}", delta_color="inverse")
        st.markdown("---")
        if current_prob > 0.7:
            st.error(f"⚠️ **High Risk**: ${exp_loss:,.0f} at risk")
        else:
            st.success(f"✅ **Secure**: ${net_val:,.0f} likely revenue")

    # 2. DETAILED REPORT (Smart Text)
    st.markdown("### 📋 Risk Assessment Report")
    
    if current_prob > 0.7:
        css_class = "risk-box-critical"
        title = "CRITICAL RISK DETECTED"
        desc = "This booking has an extremely high likelihood of cancellation."
        action = "Do not confirm without a **Non-Refundable Deposit**. Contact guest immediately to verify intent."
    elif current_prob > 0.4:
        css_class = "risk-box-moderate"
        title = "MODERATE RISK WARNING"
        desc = "Booking shows signs of volatility based on lead time and market segment."
        action = "Initiate automated retention sequence (Email/SMS). Monitor for changes."
    else:
        css_class = "risk-box-safe"
        title = "SECURE BOOKING"
        desc = "Booking profile matches high-commitment guests."
        action = "Auto-approve. Good candidate for room upgrades or upsells."
        
    st.markdown(f"""
    <div class="{css_class}">
        <h3 style="margin-top:0; color:inherit;">{title}</h3>
        <p><b>Analysis:</b> {desc}</p>
        <p><b>Recommended Action:</b> {action}</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. INTERACTIVE SIMULATOR (The "Interactive" Part)
    st.markdown("### 🎛️ Real-Time Simulator")
    st.caption("Adjust the sliders below to see how Risk changes instantly without reloading.")
    
    sim_col1, sim_col2 = st.columns(2)
    with st.container():
        st.markdown("""<div class="sim-container">""", unsafe_allow_html=True)
        
        # Interactive Widgets inside main area
        new_adr = st.slider("👇 Simulate Price (ADR) Adjustment", 50, 500, int(inputs['adr']))
        new_lead = st.slider("👇 Simulate Lead Time Adjustment", 0, 365, int(inputs['lead_time']))
        
        # Run Simulation
        sim_inputs = inputs.copy()
        sim_inputs['adr'] = new_adr
        sim_inputs['lead_time'] = new_lead
        
        sim_ordered = {k: sim_inputs.get(k, 0) for k in feature_names}
        sim_values = [sim_ordered[k] for k in feature_names]
        sim_prob = predict_risk(model, scaler, sim_values, feature_names)
        
        diff = (sim_prob - current_prob) * 100
        
        # Display Sim Result
        c1, c2 = st.columns([3, 1])
        c1.metric("Simulated Risk Probability", f"{sim_prob*100:.1f}%", f"{diff:+.1f}%")
        
        if diff < -5:
            c1.success("📉 Strategy Effective: Risk decreases significantly.")
        elif diff > 5:
            c1.error("📈 Warning: This change increases risk.")
            
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()