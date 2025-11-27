"""
SLEEP HEALTH DATA COLLECTOR & ADVISOR - PREMIUM UI
==================================================
Author: Data Science Team
Version: 4.2 (Smart Recommendations V2 & Markdown Fix)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PAGE CONFIGURATION & PREMIUM STYLING
# ============================================================

st.set_page_config(
    page_title="Sleep Health Advisor",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a "Vision-Full" and Attractive UI
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e0e0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Input Cards */
    .input-container {
        background-color: #1c232e;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2d3748;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .input-container h3 {
        color: #4facfe;
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 10px;
    }

    /* Styling Input Widgets */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #262d37;
        color: white;
        border: 1px solid #4a5568;
        border-radius: 8px;
    }
    
    /* Submit Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 30px;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        margin-top: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 114, 255, 0.6);
    }

    /* Result Metrics */
    div.metric-container {
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    /* Enhanced Recommendation Box */
    .rec-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-left: 6px solid #48bb78; /* Green accent */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        margin-top: 30px;
        color: #e2e8f0;
    }
    .rec-header {
        color: #48bb78;
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 15px;
    }
    .rec-item {
        margin-bottom: 15px;
        font-size: 1.05rem;
        line-height: 1.6;
        display: flex;
        align-items: start;
    }
    .rec-icon {
        margin-right: 10px;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DATA MANAGEMENT (ROBUST BACKEND STORAGE)
# ============================================================

# Use absolute path to ensure file is found in deployment environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, 'prediction_history.csv')

def init_storage():
    """Force creation of CSV if it doesn't exist"""
    if not os.path.exists(CSV_PATH):
        init_df = pd.DataFrame(columns=[
            "Timestamp", "Age", "Gender", "Occupation", "Sleep_Duration", 
            "Physical_Activity", "Stress_Level", "BMI_Category", "Daily_Steps", 
            "Sleep_Efficiency_Input", "Heart_Rate", "Systolic_BP", "Diastolic_BP", 
            "Calculated_Health_Risk", "Predicted_Sleep_Quality", "Predicted_Disorder", 
            "Model_Confidence", "Risk_Level"
        ])
        try:
            init_df.to_csv(CSV_PATH, index=False)
            return True
        except Exception as e:
            st.error(f"Storage Initialization Failed: {e}")
            return False
    return True

# Initialize on load
init_storage()

def save_prediction_history(data_dict):
    """Save data to Local CSV with robust error handling"""
    # Re-check/Create file just in case
    init_storage()
    
    new_row = pd.DataFrame([data_dict])
    
    try:
        # Read existing data
        existing_df = pd.read_csv(CSV_PATH)
        # Append new data
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        # Save back
        updated_df.to_csv(CSV_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Critical Error saving data: {e}")
        return False

# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """Loads models and feature lists."""
    models = {}
    
    def get_path(filename):
        return os.path.join(CURRENT_DIR, filename)

    try:
        # Load Models with absolute paths
        path_q = get_path('sleep_quality_model.pkl')
        path_d = get_path('sleep_disorder_model.pkl')
        path_e = get_path('disorder_label_encoder.pkl')

        if os.path.exists(path_q):
            with open(path_q, 'rb') as f: models['quality'] = pickle.load(f)
        else: return None
        
        if os.path.exists(path_d):
            with open(path_d, 'rb') as f: models['disorder'] = pickle.load(f)
                
        if os.path.exists(path_e):
            with open(path_e, 'rb') as f: models['encoder'] = pickle.load(f)
        
        # Load Feature Names (Critical for model compatibility)
        path_fq = get_path('feature_names_quality.csv')
        if os.path.exists(path_fq):
             models['features_quality'] = pd.read_csv(path_fq)['feature'].tolist()
        elif 'quality' in models and hasattr(models['quality'], 'feature_names_in_'):
             models['features_quality'] = models['quality'].feature_names_in_.tolist()
        else:
             st.warning("Feature list for Quality Model missing. Prediction may fail.")
             
        path_fd = get_path('feature_names_disorder.csv')
        if os.path.exists(path_fd):
             models['features_disorder'] = pd.read_csv(path_fd)['feature'].tolist()
        elif 'disorder' in models and hasattr(models['disorder'], 'feature_names_in_'):
             models['features_disorder'] = models['disorder'].feature_names_in_.tolist()
        else:
             st.warning("Feature list for Disorder Model missing. Prediction may fail.")
            
        return models
    except Exception as e:
        st.error(f"Model Loading Failed: {e}")
        return None

# ============================================================
# 4. LOGIC HELPERS (Enhanced & Fixed Recommendations)
# ============================================================

def categorize_bp(systolic, diastolic):
    if systolic < 120 and diastolic < 80: return 'Normal'
    elif systolic < 130 and diastolic < 80: return 'Elevated'
    elif systolic < 140 or diastolic < 90: return 'High_Stage1'
    else: return 'High_Stage2'

def categorize_age(age):
    if age < 30: return 'Young_Adult'
    elif age < 45: return 'Middle_Age'
    elif age < 60: return 'Senior'
    else: return 'Elderly'

def categorize_steps(steps):
    if steps < 5000: return 'Sedentary'
    elif steps < 7500: return 'Low_Active'
    elif steps < 10000: return 'Somewhat_Active'
    else: return 'Active'

def categorize_heart_rate(hr):
    if hr < 60: return 'Low'
    elif hr <= 100: return 'Normal'
    else: return 'High'

def get_smart_recommendations(inputs, disorder, sleep_quality):
    """
    Generates context-aware, smart recommendations using compound conditions.
    NOTE: Removed all internal Markdown (**bold**) from strings to fix the rendering issue.
    """
    recs = []
    
    # Extract Key Variables
    stress = inputs['Stress_Level']
    hr = inputs['Heart_Rate']
    steps = inputs['Daily_Steps']
    duration = inputs['Sleep_Duration']
    efficiency = inputs['Sleep_Efficiency_Input']
    bmi = inputs['BMI_Category']
    sys_bp = inputs['Systolic_BP']

    # --- Smarter Condition-Based Recommendations ---
    
    # 1. High Stress & Physiological Arousal (Smarter Combination)
    if stress >= 7 and hr > 75:
        recs.append("🧠 Stress/Heart Rate: High cognitive and physical arousal. Implement a **15-minute relaxation ritual** (like 4-7-8 breathing or progressive muscle relaxation) immediately before bed.")

    # 2. Low Activity & Low Sleep Quality (Smarter Combination)
    if steps < 5000 and sleep_quality < 6.5:
        recs.append("🚶 Activity & Quality: Sedentary behavior severely impacts deep sleep. Ensure you get **30 minutes of natural daylight exposure** every morning to help regulate your sleep-wake cycle.")

    # 3. Low Efficiency (Clear CBT-I Principle)
    if efficiency < 80 and duration > 7.0:
        recs.append("⏳ Sleep Efficiency: Low time spent actually sleeping while in bed. Practice **Stimulus Control**—only go to bed when sleepy, and get out if awake for more than 20 minutes.")

    # 4. Cardiovascular/Airway Risk (Smarter, Multi-factor Risk)
    high_risk_bp = sys_bp >= 140
    obese_risk = bmi == "Obese"
    
    if obese_risk and high_risk_bp:
        recs.append("🫀 Metabolic Risk: Elevated BP and Obese BMI increases risk for sleep-disordered breathing. Consult your physician immediately for a comprehensive **sleep study and weight management plan**.")

    # 5. Short Sleep Duration
    if duration < 6.0:
        recs.append("⏰ Duration Protocol: Your sleep is severely restricted. Work to extend your total time in bed by **15 minutes every week** until you reach 7-9 hours, prioritizing schedule consistency.")

    # 6. Occupation/Screen Exposure (Passive Advice)
    tech_jobs = ["Software Engineer", "Scientist", "Accountant", "Manager"]
    if inputs['Occupation'] in tech_jobs:
        recs.append("💻 Digital Detox: High screen time detected. Use blue-light filters and stop using all backlit screens **90 minutes before bed**.")
        
    # --- Disorder-Specific Protocols (Fixed Markdown Issue) ---
    
    if disorder == "Insomnia":
        recs.append("🚨 Insomnia Protocol: Focus on **CBT-I** principles, the gold standard for long-term improvement. Specifically, target correcting your association between the bed and wakefulness.")
    
    elif disorder == "Sleep Apnea":
        recs.append("🌬️ Sleep Apnea Risk: High probability of airway obstruction. Consult a sleep specialist immediately to discuss diagnostic testing (polysomnography) and treatment options like CPAP.")
        
    # --- Fallback/Maintenance ---
    if not recs:
        recs.append("✅ Excellent Sleep Metrics: Your sleep health and lifestyle indicators are optimal. Focus on maintaining a consistent bedtime and wake-up schedule, even on holidays.")

    return recs

# ============================================================
# 5. MAIN APP LOGIC
# ============================================================

def main():
    # --- Custom Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🌙 Sleep Health Advisor</h1>
        <p>Clinical-Grade Analysis & Personalized Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    models = load_models()
    if not models:
        st.error("⚠️ System Error: Model files not found or failed to load. Please verify deployment files (`.pkl` and `.csv`).")
        return

    with st.form("data_entry_form"):
        
        # --- SECTION 1: PROFILE ---
        st.markdown('<div class="input-container"><h3>👤 Identity & Profile</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Biological Age", 10, 100, 30)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
            occupation = st.selectbox("Primary Occupation", [
                "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
                "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
                "Salesperson", "Manager", "Office Worker", "Student", "Retired", "Other"
            ])
        with col2:
            bmi_category = st.selectbox("BMI Classification", ["Normal", "Overweight", "Obese", "Underweight"])
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, help="1 = Low Stress, 10 = High Stress")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION 2: LIFESTYLE ---
        st.markdown('<div class="input-container"><h3>🏃 Lifestyle & Habits</h3>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            sleep_duration = st.slider("Avg Sleep Duration (Hours)", 4.0, 12.0, 7.0, 0.1)
            daily_steps = st.number_input("Daily Steps", 0, 30000, 6000, 500)
        with col4:
            physical_activity = st.slider("Physical Activity (mins/day)", 0, 120, 45)
            sleep_efficiency_input = st.slider("Sleep Efficiency (%)", 50, 100, 85, help="% of time in bed spent sleeping") 
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION 3: BIOMETRICS ---
        st.markdown('<div class="input-container"><h3>❤️ Clinical Biometrics</h3>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5: heart_rate = st.number_input("Resting HR (bpm)", 40, 120, 70)
        with col6: sys_bp = st.number_input("Systolic BP (mmHg)", 90, 180, 120)
        with col7: dia_bp = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)
        st.markdown('</div>', unsafe_allow_html=True)

        submit_btn = st.form_submit_button("🚀 ANALYZE & GET INSIGHTS")

    if submit_btn:
        try:
            # --- BACKEND PROCESSING ---
            
            # 1. Feature Engineering
            bp_cat = categorize_bp(sys_bp, dia_bp)
            age_grp = categorize_age(age)
            step_cat = categorize_steps(daily_steps)
            hr_cat = categorize_heart_rate(heart_rate)
            
            bmi_val = {'Normal': 1, 'Underweight': 0, 'Overweight': 2, 'Obese': 3}.get(bmi_category, 1)
            # Adjusted calculation to reflect health risk factors
            health_score = (stress_level * 0.4) + ((10 - physical_activity/10) * 0.2) + ((heart_rate/70) * 0.2) + (bmi_val * 0.2)
            sleep_eff = (sleep_duration * sleep_efficiency_input / 100)

            # 2. Feature Vector Construction for Quality Model
            cols_q = models.get('features_quality', [])
            input_df_q = pd.DataFrame(0, index=[0], columns=cols_q)
            
            # Numericals
            vals = {'Age': age, 'Sleep Duration': sleep_duration, 'Physical Activity Level': physical_activity,
                    'Stress Level': stress_level, 'Heart Rate': heart_rate, 'Daily Steps': daily_steps,
                    'Systolic_BP': sys_bp, 'Diastolic_BP': dia_bp, 'Sleep_Efficiency': sleep_eff, 'Health_Risk_Score': health_score}
            for k, v in vals.items():
                if k in input_df_q: input_df_q[k] = v

            # Ordinals
            bmi_m = {'Underweight': 0, 'Normal': 1, 'Normal Weight': 1, 'Overweight': 2, 'Obese': 3}
            bp_m = {'Normal': 0, 'Elevated': 1, 'High_Stage1': 2, 'High_Stage2': 3}
            
            act_cat = 0 if physical_activity < 30 else (1 if physical_activity < 60 else 2)
            str_cat = 0 if stress_level <= 3 else (1 if stress_level <= 6 else 2)
            dur_cat = 0 if sleep_duration < 6 else (1 if sleep_duration < 7 else (2 if sleep_duration <= 9 else 3))

            ords = {'BMI Category_Encoded': bmi_m.get(bmi_category, 1), 'BP_Category_Encoded': bp_m.get(bp_cat, 0),
                    'Activity_Category_Encoded': act_cat, 'Stress_Category_Encoded': str_cat, 'Sleep_Duration_Category_Encoded': dur_cat}
            for k, v in ords.items():
                if k in input_df_q: input_df_q[k] = v

            # One-Hot
            targets = [f"Gender_{gender}", f"Occupation_{occupation}", f"Age_Group_{age_grp}", 
                        f"Heart_Rate_Category_{hr_cat}", f"Steps_Category_{step_cat}", f"BP_Category_{bp_cat}"]
            for t in targets:
                if t in input_df_q: input_df_q[t] = 1

            # 3. Model Prediction - Sleep Quality
            pred_qual = models['quality'].predict(input_df_q)[0]
            
            # 3. Model Prediction - Sleep Disorder
            if 'features_disorder' in models:
                # Align features specifically for the disorder model
                cols_d = models['features_disorder']
                input_df_d = pd.DataFrame(0, index=[0], columns=cols_d)
                
                # Transfer common columns
                com_cols = list(set(input_df_q.columns) & set(input_df_d.columns))
                input_df_d[com_cols] = input_df_q[com_cols]
                final_in = input_df_d.reindex(columns=cols_d, fill_value=0) # Ensure column order
            else:
                final_in = input_df_q

            pred_idx = models['disorder'].predict(final_in)[0]
            pred_dis = models['encoder'].inverse_transform([pred_idx])[0] if 'encoder' in models else "Unknown"
            
            try: conf = np.max(models['disorder'].predict_proba(final_in)) * 100
            except: conf = 0.0
            
            risk = "Low Risk" if pred_dis == "None" and pred_qual >= 6.5 else "Moderate Risk"
            if pred_dis != "None" and conf > 75: risk = "High Risk"

            # 4. Save to Backend (Include Prediction Results)
            success = save_prediction_history({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Age": age, "Gender": gender, "Occupation": occupation,
                "Sleep_Duration": sleep_duration, "Physical_Activity": physical_activity,
                "Stress_Level": stress_level, "BMI_Category": bmi_category,
                "Daily_Steps": daily_steps, "Sleep_Efficiency_Input": sleep_efficiency_input,
                "Heart_Rate": heart_rate, "Systolic_BP": sys_bp, "Diastolic_BP": dia_bp,
                "Calculated_Health_Risk": round(health_score, 2),
                "Predicted_Sleep_Quality": round(pred_qual, 2),
                "Predicted_Disorder": pred_dis,
                "Model_Confidence": round(conf, 2),
                "Risk_Level": risk
            })
            
            if success:
                st.success("✅ Analysis Complete & Data Logged!")
                
                # 5. Frontend Recommendations (Smart & Fixed)
                tips_list = get_smart_recommendations({
                    'Occupation': occupation, 'Stress_Level': stress_level,
                    'Sleep_Duration': sleep_duration, 'Sleep_Efficiency_Input': sleep_efficiency_input,
                    'BMI_Category': bmi_category, 'Systolic_BP': sys_bp, 'Daily_Steps': daily_steps,
                    'Physical_Activity': physical_activity, 'Heart_Rate': heart_rate
                }, pred_dis, pred_qual)

                # Format tips HTML for cleaner rendering
                formatted_tips = ""
                for tip in tips_list:
                    # Note: No Markdown like ** or * is used within the tip string itself
                    formatted_tips += f'<div class="rec-item"><span class="rec-icon">👉</span><span>{tip}</span></div>'
                
                # Display prediction cards
                st.markdown("### 🧬 Analysis Results")
                r1, r2, r3 = st.columns(3)
                
                with r1:
                    st.markdown(f"""<div class="metric-container"><div style="color:#9ca3af;">Sleep Quality (1-10)</div><div style="font-size:3.5em; color:#f3f4f6;">{pred_qual:.1f}</div></div>""", unsafe_allow_html=True)
                with r2:
                    col = "#10b981" if pred_dis == "None" else ("#f59e0b" if risk == "Moderate Risk" else "#ef4444")
                    icon = "🛡️" if pred_dis == "None" else "⚠️"
                    st.markdown(f"""<div class="metric-container" style="border-bottom: 4px solid {col};"><div style="color:#9ca3af;">Diagnosis ({risk})</div><div style="font-size:1.8em; color:{col}; margin-top:10px;">{icon} {pred_dis}</div></div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""<div class="metric-container"><div style="color:#9ca3af;">Model Certainty</div><div style="font-size:3.5em; color:#f3f4f6;">{conf:.1f}%</div></div>""", unsafe_allow_html=True)
                
                # Display Recommendation Box
                st.markdown(f"""
                <div class="rec-container">
                    <div class="rec-header">🩺 Personalized Action Plan</div>
                    <p style="font-size: 1.1rem; margin-bottom: 20px; color: #cbd5e0;">
                        Based on your bio-markers and lifestyle, here is your tailored wellness strategy:
                    </p>
                    {formatted_tips}
                    <div style="margin-top: 20px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); color: #a0aec0; font-size: 0.9rem;">
                        <i>*Disclaimer: This is an AI-based analysis, not a medical diagnosis. Consult a doctor for any health concerns.</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error("Submission failed. Please try again.")

        except Exception as e:
            st.error(f"Processing Error: An unexpected error occurred during prediction: {e}")

if __name__ == "__main__":
    main()