"""
SLEEP HEALTH & DISORDER PREDICTION - INTELLIGENT STREAMLIT APP
===============================================================
Author: Data Science Team
Version: 13.1 (Fixing Recommendation Display)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Sleep Health AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Ultimate UI/UX
st.markdown("""
    <style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .main {
        background-color: #050505; /* Deepest Black */
        font-family: 'Inter', sans-serif;
        background-image: radial-gradient(circle at 50% 50%, #111827 0%, #050505 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }

    /* Input Cards - Glassmorphism Style */
    .input-card {
        background: rgba(30, 30, 30, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .input-card:hover {
        border-color: rgba(88, 166, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Section Headers inside Cards */
    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
    }
    .card-icon {
        font-size: 1.5rem;
        margin-right: 10px;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Advanced Metric Result Cards */
    div.metric-container {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    div.metric-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    div.metric-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px -10px rgba(59, 130, 246, 0.3);
    }

    /* Custom Submit Button */
    .stButton>button {
        background: linear-gradient(92.88deg, #455EB5 9.16%, #5643CC 43.89%, #673FD7 64.72%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        height: 65px;
        width: 100%;
        font-size: 1.2rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 5px 15px rgba(86, 67, 204, 0.4);
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(86, 67, 204, 0.6);
        transform: scale(1.02);
    }

    /* Input Field Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    .stSlider div[data-baseweb="slider"] {
        /* Color for slider track */
    }
    
    /* Recommendations Styling */
    .rec-box {
        background-color: #0d1117;
        border-left: 4px solid #238636;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. UTILITY FUNCTIONS & MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """Load ML models and encoders with error handling"""
    models = {}
    try:
        # Load Quality Model (Regression)
        if os.path.exists('sleep_quality_model.pkl'):
            with open('sleep_quality_model.pkl', 'rb') as f:
                models['quality'] = pickle.load(f)
        
        # Load Disorder Model (Classification)
        if os.path.exists('sleep_disorder_model.pkl'):
            with open('sleep_disorder_model.pkl', 'rb') as f:
                models['disorder'] = pickle.load(f)
                
        # Load Label Encoder
        if os.path.exists('disorder_label_encoder.pkl'):
            with open('disorder_label_encoder.pkl', 'rb') as f:
                models['encoder'] = pickle.load(f)
        
        # Feature Names - Try to get from model object first (most reliable)
        if 'quality' in models and hasattr(models['quality'], 'feature_names_in_'):
             models['features_quality'] = models['quality'].feature_names_in_.tolist()
        elif os.path.exists('feature_names_quality.csv'):
            models['features_quality'] = pd.read_csv('feature_names_quality.csv')['feature'].tolist()

        if 'disorder' in models and hasattr(models['disorder'], 'feature_names_in_'):
             models['features_disorder'] = models['disorder'].feature_names_in_.tolist()
        elif os.path.exists('feature_names_disorder.csv'):
            models['features_disorder'] = pd.read_csv('feature_names_disorder.csv')['feature'].tolist()
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def save_prediction_history(data_dict):
    file_path = 'prediction_history.csv'
    new_row = pd.DataFrame([data_dict])
    if not os.path.exists(file_path):
        new_row.to_csv(file_path, index=False)
    else:
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(file_path, index=False)
        except Exception:
            new_row.to_csv(file_path, index=False)

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

def get_smart_recommendations(inputs, disorder):
    recs = []
    
    # Occupation Logic
    tech_jobs = ["Software Engineer", "Scientist", "Accountant", "Manager", "Office Worker"]
    active_jobs = ["Nurse", "Doctor", "Construction Worker", "Firefighter", "Police Officer"]
    
    if inputs['Occupation'] in tech_jobs:
        recs.append("💻 **Digital Detox:** High screen time detected. Use blue-light blockers and stop screens 90m before bed.")
    elif inputs['Occupation'] in active_jobs:
        recs.append("🏥 **Physical Recovery:** High-demand job. Ensure mattress supports spinal alignment and prioritize cool room temp.")
    
    # Stress Logic
    if inputs['Stress_Level'] >= 7:
        recs.append("🧘 **Cortisol Management:** High stress delays sleep. Try '4-7-8 Breathing' immediately upon getting into bed.")

    # Efficiency Logic
    if inputs['Sleep_Efficiency_Input'] < 80:
        recs.append("⏳ **Sleep Restriction:** Too much time awake in bed. Go to bed *only* when tired to reset sleep drive.")

    # BMI/BP Logic
    if inputs['BMI_Category'] in ["Overweight", "Obese"] or inputs['Systolic_BP'] > 130:
        recs.append("🫀 **Cardio Health:** Elevated markers correlate with airway obstruction. Try side-sleeping for immediate relief.")

    # Activity Logic
    if inputs['Daily_Steps'] < 5000:
        recs.append("🚶 **Activity Boost:** Sedentary lifestyle affects deep sleep. A 20-min morning walk sets circadian rhythms.")

    # Disorder Logic
    if disorder == "Insomnia" and len(recs) < 3:
        recs.append("🛌 **CBT-I:** Consider Cognitive Behavioral Therapy for Insomnia, the gold standard for long-term improvement.")
    elif disorder == "Sleep Apnea" and len(recs) < 3:
        recs.append("🌬️ **Airway Check:** Consult a specialist about CPAP or dental devices.")

    # Return as a clean list of strings, we will join them later with double newlines
    return recs

# ============================================================
# 3. MAIN APPLICATION LOGIC
# ============================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<div style='text-align: center; font-size: 5em; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(139, 92, 246, 0.5));'>🌙</div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #f0f6fc; margin: 0; font-weight: 800; letter-spacing: 1px;'>SleepAI</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e; font-size: 0.8em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 30px;'>Intelligence System</p>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation", 
            ["🚀 Diagnostics", "📊 Analytics", "ℹ️ Methodology"],
            index=0
        )
        
        st.markdown("---")
        with st.expander("💡 **Quick Tips**", expanded=True):
            st.caption("• Measure BP at rest")
            st.caption("• Use 7-day averages")
            st.caption("• Update monthly")
            
        st.markdown("<div style='margin-top: 50px; text-align: center; color: #484f58; font-size: 0.8em;'>v13.1 • Pro Edition</div>", unsafe_allow_html=True)

    models = load_models()
    
    if page == "🚀 Diagnostics":
        render_prediction_page(models)
    elif page == "📊 Analytics":
        render_history_page()
    else:
        render_about_page()

# ============================================================
# 4. PREDICTION PAGE
# ============================================================

def render_prediction_page(models):
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 3em; margin-bottom: 10px; background: -webkit-linear-gradient(0deg, #ffffff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Intelligent Sleep Diagnostics
        </h1>
        <p style="color: #9ca3af; font-size: 1.2em; max-width: 600px; margin: 0 auto;">
            Clinical-grade AI analysis of your biomarkers to predict Sleep Quality and Disorder Risks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        
        # --- SECTION 1: PROFILE ---
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-icon">👤</span> Identity & Profile
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            age = st.number_input("Biological Age", 10, 100, 30)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with col2:
            occupation = st.selectbox("Primary Occupation", [
                "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
                "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
                "Salesperson", "Manager", "Office Worker", "Student", "Retired",
                "Artist", "Other"
            ])
            bmi_category = st.selectbox("BMI Classification", ["Normal", "Overweight", "Obese", "Underweight"])
        
        st.markdown("</div>", unsafe_allow_html=True)

        # --- SECTION 2: LIFESTYLE ---
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-icon">🏃</span> Lifestyle & Habits
            </div>
        """, unsafe_allow_html=True)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            sleep_duration = st.slider("Avg Sleep (Hours)", 4.0, 12.0, 7.0, 0.1)
            daily_steps = st.number_input("Daily Steps", 0, 30000, 6000, 500)
        with col4:
            physical_activity = st.slider("Activity (mins/day)", 0, 120, 45)
            sleep_efficiency_input = st.slider("Sleep Efficiency (%)", 50, 100, 85)
        with col5:
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # --- SECTION 3: BIOMETRICS ---
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-icon">❤️</span> Clinical Biometrics
            </div>
        """, unsafe_allow_html=True)
        
        b1, b2, b3 = st.columns(3)
        with b1:
            heart_rate = st.number_input("Resting HR (bpm)", 40, 120, 70)
        with b2:
            sys_bp = st.number_input("Systolic BP (mmHg)", 90, 180, 120)
        with b3:
            dia_bp = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # Submit Button
        submit_btn = st.form_submit_button("🚀 GENERATE ANALYSIS")

    # --- LOGIC ---
    if submit_btn:
        if not models or 'quality' not in models:
            st.error("⚠️ Models unavailable.")
            return

        with st.spinner("💡 Analyzing bio-markers..."):
            try:
                # 1. Feature Engineering
                bp_category = categorize_bp(sys_bp, dia_bp)
                age_group = categorize_age(age)
                steps_cat = categorize_steps(daily_steps)
                hr_cat = categorize_heart_rate(heart_rate)
                
                bmi_risk_val = {'Normal': 1, 'Underweight': 1, 'Overweight': 2, 'Obese': 3}.get(bmi_category, 1)
                health_risk_score = (
                    (stress_level * 0.3) + 
                    ((10 - physical_activity/10) * 0.3) + 
                    (heart_rate/10 * 0.2) + 
                    (bmi_risk_val * 0.2)
                )
                
                sleep_eff_val = (sleep_duration * sleep_efficiency_input / 100)

                # 2. Feature Construction
                expected_cols = models['features_quality']
                input_df = pd.DataFrame(0, index=[0], columns=expected_cols)
                
                # Numericals
                input_df['Age'] = age
                input_df['Sleep Duration'] = sleep_duration
                input_df['Physical Activity Level'] = physical_activity
                input_df['Stress Level'] = stress_level
                input_df['Heart Rate'] = heart_rate
                input_df['Daily Steps'] = daily_steps
                input_df['Systolic_BP'] = sys_bp
                input_df['Diastolic_BP'] = dia_bp
                input_df['Sleep_Efficiency'] = sleep_eff_val
                input_df['Health_Risk_Score'] = health_risk_score

                # Ordinals (Manual Map)
                bmi_map = {'Underweight': 0, 'Normal': 1, 'Normal Weight': 1, 'Overweight': 2, 'Obese': 3}
                bp_map = {'Normal': 0, 'Elevated': 1, 'High_Stage1': 2, 'High_Stage2': 3}
                
                if physical_activity < 30: act_cat = 0 
                elif physical_activity < 60: act_cat = 1 
                else: act_cat = 2 

                if stress_level <= 3: stress_cat = 0 
                elif stress_level <= 6: stress_cat = 1 
                else: stress_cat = 2 
                    
                if sleep_duration < 6: dur_cat = 0
                elif sleep_duration < 7: dur_cat = 1
                elif sleep_duration <= 9: dur_cat = 2
                else: dur_cat = 3

                input_df['BMI Category_Encoded'] = bmi_map.get(bmi_category, 1)
                input_df['BP_Category_Encoded'] = bp_map.get(bp_category, 0)
                input_df['Activity_Category_Encoded'] = act_cat
                input_df['Stress_Category_Encoded'] = stress_cat
                input_df['Sleep_Duration_Category_Encoded'] = dur_cat

                # One-Hot
                targets = [
                    f"Gender_{gender}", f"Occupation_{occupation}", 
                    f"Age_Group_{age_group}", f"Heart_Rate_Category_{hr_cat}", 
                    f"Steps_Category_{steps_cat}", f"BP_Category_{bp_category}"
                ]
                for t in targets:
                    if t in input_df.columns: input_df[t] = 1

                # 3. Prediction
                pred_quality = models['quality'].predict(input_df)[0]
                
                # Align for disorder model if needed
                if 'features_disorder' in models:
                    input_df_disorder = pd.DataFrame(0, index=[0], columns=models['features_disorder'])
                    common = list(set(input_df.columns) & set(input_df_disorder.columns))
                    input_df_disorder[common] = input_df[common]
                    for t in targets:
                        if t in input_df_disorder.columns: input_df_disorder[t] = 1
                    final_input = input_df_disorder
                else:
                    final_input = input_df

                pred_disorder_idx = models['disorder'].predict(final_input)[0]
                
                if 'encoder' in models:
                    pred_disorder = models['encoder'].inverse_transform([pred_disorder_idx])[0]
                else:
                    pred_disorder = "Unknown"
                
                try:
                    conf = np.max(models['disorder'].predict_proba(final_input)) * 100
                except:
                    conf = 0.0
                
                if conf < 50: risk_level = "Low Confidence"
                elif pred_disorder == "None": risk_level = "Low Risk"
                else: risk_level = "High Risk"

                # --- RESULTS DISPLAY ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🧬 Analysis Results")
                
                r1, r2, r3 = st.columns(3)
                
                # Quality Card
                with r1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Sleep Quality Score</div>
                        <div style="font-size: 3.5rem; font-weight: 800; color: #f3f4f6; text-shadow: 0 0 20px rgba(255,255,255,0.1);">{pred_quality:.1f}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Clinical Index (0-10)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Diagnosis Card
                with r2:
                    status_color = "#10b981" if pred_disorder == "None" else "#ef4444"
                    icon = "🛡️" if pred_disorder == "None" else "⚠️"
                    st.markdown(f"""
                    <div class="metric-container" style="border-bottom: 4px solid {status_color};">
                        <div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Risk Assessment</div>
                        <div style="font-size: 2rem; font-weight: 700; color: {status_color}; margin-top: 10px;">{icon} {pred_disorder}</div>
                        <div style="font-size: 0.8rem; color: #6b7280; margin-top: 5px;">Primary Detection</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Card
                with r3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Model Certainty</div>
                        <div style="font-size: 3.5rem; font-weight: 800; color: #f3f4f6;">{conf:.1f}%</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Confidence Interval</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- DEEP INSIGHTS ---
                st.markdown("### 🔍 Deep Insights")
                di1, di2 = st.columns(2)
                
                with di1:
                    st.info(f"**Health Risk Score:** {health_risk_score:.2f} (Normalized Metric)")
                with di2:
                    delta = pred_quality - 7.2
                    st.metric("vs. Population Avg", f"{pred_quality:.1f}", f"{delta:.1f}", delta_color="normal")

                # --- RECOMMENDATIONS ---
                smart_tips = get_smart_recommendations({
                    'Occupation': occupation, 'Stress_Level': stress_level,
                    'Sleep_Duration': sleep_duration, 'Sleep_Efficiency_Input': sleep_efficiency_input,
                    'BMI_Category': bmi_category, 'Systolic_BP': sys_bp, 'Daily_Steps': daily_steps
                }, pred_disorder)

                # Format recommendations into a bulleted string with double newlines for proper Markdown rendering
                formatted_tips = ""
                for tip in smart_tips:
                    formatted_tips += f"- {tip}\n\n"

                st.subheader("🩺 Personalized Action Plan")
                
                if pred_disorder != "None":
                    st.warning(f"### *⚠️ High Risk Detected:* {pred_disorder} \n \n *Analysis:* Our model flagged this risk primarily based on your reported  *stress level ({stress_level}/10)* and *sleep efficiency ({sleep_efficiency_input}%)*. \n --- \n  **📋 Your Tailored Action Plan:** \n \n {formatted_tips} ")
                else:
                    st.success(f"### ✅ Optimal Health Detected \n \n **Analysis:** Your bio-markers indicate a balanced circadian rhythm and healthy lifestyle habits. \n --- \n  \n **📋 Wellness Tips to Maintain:** \n \n {formatted_tips}")

                # --- SAVE ---
                save_prediction_history({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age, "Gender": gender, "Occupation": occupation,
                    "Sleep_Duration": sleep_duration, "Physical_Activity": physical_activity,
                    "Stress_Level": stress_level, "BMI_Category": bmi_category,
                    "Daily_Steps": daily_steps, "Sleep_Efficiency_Input": sleep_efficiency_input,
                    "Heart_Rate": heart_rate, "Systolic_BP": sys_bp, "Diastolic_BP": dia_bp,
                    "Calculated_Health_Risk": round(health_risk_score, 2),
                    "Predicted_Sleep_Quality": round(pred_quality, 2),
                    "Predicted_Disorder": pred_disorder,
                    "Model_Confidence": round(conf, 2),
                    "Risk_Level": risk_level
                })
                st.toast("Data securely logged.", icon="🔒")

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 5. HISTORY PAGE
# ============================================================

def render_history_page():
    st.title("📊 Analytics Dashboard")
    if not os.path.exists('prediction_history.csv'):
        st.info("No data yet.")
        return

    df = pd.read_csv('prediction_history.csv')
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Scans", len(df))
    if 'Predicted_Sleep_Quality' in df.columns:
        k2.metric("Avg Quality", f"{df['Predicted_Sleep_Quality'].mean():.1f}")
    if 'Predicted_Disorder' in df.columns:
        risks = df[df['Predicted_Disorder'] != 'None'].shape[0]
        k3.metric("Risks Flagged", risks)
    if 'Age' in df.columns:
        k4.metric("Avg Age", f"{df['Age'].mean():.0f}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Quality Trend")
        if 'Timestamp' in df.columns:
            fig = px.line(df, x='Timestamp', y='Predicted_Sleep_Quality', markers=True, line_shape='spline')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.markdown("##### Disorder Distribution")
        if 'Predicted_Disorder' in df.columns:
            fig = px.pie(df, names='Predicted_Disorder', hole=0.6, color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b'])
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗃️ Raw Data Log")
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)

# ============================================================
# 6. ABOUT PAGE
# ============================================================

def render_about_page():
    st.title("ℹ️ About SleepAI")
    st.markdown("### Clinical-Grade Intelligence\nPowered by XGBoost & Random Forest algorithms.")

if __name__ == "__main__":
    main()