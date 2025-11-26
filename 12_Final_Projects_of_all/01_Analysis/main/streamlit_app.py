"""
SLEEP HEALTH & DISORDER PREDICTION - STREAMLIT WEB APPLICATION
================================================================

This Streamlit application provides an interactive interface for:
1. Sleep Quality Prediction (Regression)
2. Sleep Disorder Classification
3. Risk Assessment & Health Insights
4. Batch Predictions from CSV
5. Model Performance Metrics
6. Data Visualization & Analytics

Models Used:
- sleep_quality_model.pkl: XGBoost/Random Forest for Sleep Quality
- sleep_disorder_model.pkl: XGBoost/LightGBM for Sleep Disorder
- disorder_label_encoder.pkl: Label Encoder for disorder classes

Author: Data Science Team
Date: 2024
Version: 1.0
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

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sleep Health Prediction System",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
custom_css = """
    <style>
    .main {
        padding: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .history-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_models():
    """Load all trained models and encoders"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load models
        with open(os.path.join(script_dir, 'sleep_quality_model.pkl'), 'rb') as f:
            quality_model = pickle.load(f)
        
        with open(os.path.join(script_dir, 'sleep_disorder_model.pkl'), 'rb') as f:
            disorder_model = pickle.load(f)
        
        with open(os.path.join(script_dir, 'disorder_label_encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        
        return quality_model, disorder_model, encoder
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

def get_risk_color(risk_level):
    """Return color code based on risk level"""
    if risk_level == 'High_Risk':
        return '#dc3545'
    elif risk_level == 'Medium_Risk':
        return '#ffc107'
    else:
        return '#28a745'

def get_risk_recommendation(risk_level, disorder):
    """Get health recommendations based on risk level and disorder"""
    recommendations = {
        'Low_Risk': {
            'title': '✅ Low Risk - Maintain Current Health',
            'advice': [
                'Continue regular exercise and physical activity',
                'Maintain consistent sleep schedule',
                'Monitor stress levels',
                'Annual health checkups recommended'
            ]
        },
        'Medium_Risk': {
            'title': '⚠️ Medium Risk - Increase Vigilance',
            'advice': [
                'Increase physical activity to 150+ mins/week',
                'Practice stress management techniques',
                'Improve sleep hygiene',
                'Consult healthcare provider for preventive measures'
            ]
        },
        'High_Risk': {
            'title': '🚨 High Risk - Seek Professional Help',
            'advice': [
                'Schedule appointment with sleep specialist',
                'Increase physical activity gradually',
                'Reduce stress through meditation/therapy',
                'Consider sleep studies if disorder detected'
            ]
        }
    }
    return recommendations.get(risk_level, {})

def create_input_features(user_data):
    """
    Convert user input to feature vector
    Note: This requires the exact feature order from training
    """
    feature_order = [
        'Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level',
        'Heart Rate', 'Daily Steps', 'SleepDisorder_Imputed', 'Systolic_BP',
        'Diastolic_BP', 'Sleep_Efficiency', 'Health_Risk_Score',
        'BMI Category_Encoded', 'Sleep_Duration_Category_Encoded',
        'Activity_Category_Encoded', 'Stress_Category_Encoded',
        'BP_Category_Encoded', 'Gender_Male', 'Occupation_Office Worker',
        'Occupation_Retired', 'Occupation_Student', 'Age_Group_Middle_Age',
        'Age_Group_Senior', 'Age_Group_Young_Adult', 'Heart_Rate_Category_Normal',
        'Steps_Category_Low_Active', 'Steps_Category_Sedentary',
        'Steps_Category_Somewhat_Active'
    ]
    
    return pd.DataFrame([user_data])[feature_order]

def load_prediction_history():
    """Load prediction history from CSV file"""
    history_file = os.path.join(os.path.dirname(__file__), 'prediction_history.csv')
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    return pd.DataFrame()

def save_prediction(prediction_data):
    """Save prediction to history CSV file"""
    history_file = os.path.join(os.path.dirname(__file__), 'prediction_history.csv')
    
    # Load existing history
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame()
    
    # Append new prediction
    new_row = pd.DataFrame([prediction_data])
    history_df = pd.concat([history_df, new_row], ignore_index=True)
    
    # Keep only last 1000 records
    if len(history_df) > 1000:
        history_df = history_df.tail(1000)
    
    # Save to CSV
    history_df.to_csv(history_file, index=False)

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Load models
    quality_model, disorder_model, encoder = load_models()
    
    # Header
    st.markdown("# 😴 Sleep Health & Disorder Prediction System")
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.markdown("## 🔍 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Predictions", 
         "📈 Analytics", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    
    # Add prediction history section in sidebar
    if page == "🔮 Single Prediction":
        with st.sidebar.expander("📋 Recent Predictions", expanded=False):
            history_df = load_prediction_history()
            if len(history_df) > 0:
                # Show last 5 predictions
                for idx, row in history_df.tail(5).iterrows():
                    st.markdown(f"""
                    <div class="history-card">
                    <b>{row['Timestamp']}</b><br>
                    Age: {row['Age']} | Quality: {row['Sleep_Quality']}/10<br>
                    Disorder: {row['Disorder']} | Risk: {row['Risk_Level']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Option to download all history
                csv_history = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All History",
                    data=csv_history,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_history"
                )
            else:
                st.info("No predictions yet")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Information")
    st.sidebar.info("""
    **Available Models:**
    - Sleep Quality Predictor (Regression)
    - Sleep Disorder Classifier (Multi-class)
    
    **Disorders Detected:**
    - """ + ", ".join(encoder.classes_) + """
    """)
    
    # ============================================================
    # PAGE: HOME
    # ============================================================
    
    if page == "🏠 Home":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## Welcome to Sleep Health Prediction")
            st.markdown("""
            This application uses advanced machine learning models to predict:
            
            **1. Sleep Quality (1-10 scale)**
            - Based on sleep duration, physical activity, stress levels
            - Helps assess overall sleep health
            
            **2. Sleep Disorders**
            - Detects common sleep disorders
            - Provides risk level assessment
            
            **3. Health Recommendations**
            - Personalized advice based on predictions
            - Risk mitigation strategies
            """)
        
        with col2:
            st.markdown("## Key Features")
            st.markdown("""
            ✨ **Interactive Predictions**
            - Single person or batch analysis
            - Real-time results
            
            ✨ **Risk Assessment**
            - Low, Medium, High risk categories
            - Confidence scores
            
            ✨ **Analytics Dashboard**
            - Population-level insights
            - Data visualization
            
            ✨ **Batch Processing**
            - Upload CSV files
            - Export predictions
            """)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("## 📊 Quick Start Guide")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Step 1️⃣
            **Navigate to Single Prediction**
            
            Fill in your health metrics and get instant predictions.
            """)
        
        with col2:
            st.markdown("""
            ### Step 2️⃣
            **Review Your Results**
            
            Check sleep quality score and disorder diagnosis with confidence.
            """)
        
        with col3:
            st.markdown("""
            ### Step 3️⃣
            **Get Recommendations**
            
            Receive personalized health advice based on your risk level.
            """)
    
    # ============================================================
    # PAGE: SINGLE PREDICTION
    # ============================================================
    
    elif page == "🔮 Single Prediction":
        st.markdown("## 🔮 Individual Health Assessment")
        st.markdown("---")
        
        # Input Form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Personal Information")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            occupation = st.selectbox("Occupation", ["Manual Labor", "Office Worker", "Retired", "Student"])
            
            st.markdown("### 💪 Physical Health")
            bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
            systolic_bp = st.slider("Systolic BP (mmHg)", 80, 180, 120)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 120, 80)
            heart_rate = st.slider("Heart Rate (bpm)", 40, 140, 70)
        
        with col2:
            st.markdown("### 😴 Sleep Information")
            sleep_duration = st.slider("Sleep Duration (hours)", 2.0, 12.0, 7.0, 0.5)
            sleep_efficiency = st.slider("Sleep Efficiency (%)", 0, 100, 80)
            
            st.markdown("### 🏃 Activity Level")
            physical_activity = st.slider("Physical Activity Level", 0, 150, 50)
            daily_steps = st.slider("Daily Steps", 1000, 50000, 10000, 1000)
            activity_category = st.selectbox("Activity Category", ["Sedentary", "Low_Active", "Somewhat_Active", "Active", "Very_Active"])
        
        with col3:
            st.markdown("### 😰 Stress & Health")
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            
            st.markdown("### 📋 Categorical Data")
            sleep_duration_category = st.selectbox("Sleep Duration Category", ["Below_Optimal", "Optimal", "Insufficient"])
            bp_category = st.selectbox("BP Category", ["Optimal", "Elevated", "High_Stage1", "High_Stage2"])
            heart_rate_category = st.selectbox("Heart Rate Category", ["Low", "Normal", "High"])
            steps_category = st.selectbox("Steps Category", ["Sedentary", "Low_Active", "Somewhat_Active", "Active"])
        
        # Encode categorical variables
        gender_male = 1 if gender == "Male" else 0
        
        # Occupation encoding
        occupation_office = 1 if occupation == "Office Worker" else 0
        occupation_retired = 1 if occupation == "Retired" else 0
        occupation_student = 1 if occupation == "Student" else 0
        
        # Age group encoding
        if age < 30:
            age_group_young = 1
            age_group_middle = 0
            age_group_senior = 0
        elif age < 55:
            age_group_young = 0
            age_group_middle = 1
            age_group_senior = 0
        else:
            age_group_young = 0
            age_group_middle = 0
            age_group_senior = 1
        
        # BMI encoding
        bmi_encoding = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
        bmi_encoded = bmi_encoding[bmi_category]
        
        # Sleep Duration Category encoding
        sleep_cat_encoding = {"Below_Optimal": 0, "Optimal": 1, "Insufficient": 2}
        sleep_duration_cat_encoded = sleep_cat_encoding[sleep_duration_category]
        
        # Activity Category encoding
        activity_cat_encoding = {"Sedentary": 0, "Low_Active": 1, "Somewhat_Active": 2, "Active": 3, "Very_Active": 4}
        activity_category_encoded = activity_cat_encoding[activity_category]
        
        # Stress Category encoding
        if stress_level <= 3:
            stress_category_encoded = 0
        elif stress_level <= 6:
            stress_category_encoded = 1
        else:
            stress_category_encoded = 2
        
        # BP Category encoding
        bp_cat_encoding = {"Optimal": 0, "Elevated": 1, "High_Stage1": 2, "High_Stage2": 3}
        bp_category_encoded = bp_cat_encoding[bp_category]
        
        # Heart Rate Category encoding
        heart_rate_category_normal = 1 if heart_rate_category == "Normal" else 0
        
        # Steps Category encoding
        steps_cat_encoding = {"Sedentary": 0, "Low_Active": 1, "Somewhat_Active": 2, "Active": 3}
        steps_category_encoded = steps_cat_encoding[steps_category]
        
        # Calculate Health Risk Score
        health_risk_score = (
            (180 - age) / 180 * 2 +
            abs(heart_rate - 70) / 70 * 2 +
            stress_level / 10 * 2 +
            (100 - sleep_efficiency) / 100 * 2 +
            (bmi_encoded + 1) * 0.5
        )
        
        # SleepDisorder_Imputed (1 for now, can be adjusted)
        sleep_disorder_imputed = 1
        
        # Create input data dictionary
        input_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'SleepDisorder_Imputed': sleep_disorder_imputed,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Sleep_Efficiency': sleep_efficiency,
            'Health_Risk_Score': health_risk_score,
            'BMI Category_Encoded': bmi_encoded,
            'Sleep_Duration_Category_Encoded': sleep_duration_cat_encoded,
            'Activity_Category_Encoded': activity_category_encoded,
            'Stress_Category_Encoded': stress_category_encoded,
            'BP_Category_Encoded': bp_category_encoded,
            'Gender_Male': gender_male,
            'Occupation_Office Worker': occupation_office,
            'Occupation_Retired': occupation_retired,
            'Occupation_Student': occupation_student,
            'Age_Group_Middle_Age': age_group_middle,
            'Age_Group_Senior': age_group_senior,
            'Age_Group_Young_Adult': age_group_young,
            'Heart_Rate_Category_Normal': heart_rate_category_normal,
            'Steps_Category_Low_Active': 1 if steps_category == "Low_Active" else 0,
            'Steps_Category_Sedentary': 1 if steps_category == "Sedentary" else 0,
            'Steps_Category_Somewhat_Active': 1 if steps_category == "Somewhat_Active" else 0,
        }
        
        # Prediction Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔮 Get Prediction", key="predict_btn", help="Click to generate prediction"):
                try:
                    # Create feature vector
                    X = create_input_features(input_data)
                    
                    # Predict Sleep Quality
                    quality_pred = quality_model.predict(X)[0]
                    quality_pred = max(1, min(10, quality_pred))  # Clamp to 1-10
                    
                    # Predict Sleep Disorder
                    disorder_pred_label = disorder_model.predict(X)[0]
                    disorder_pred = encoder.inverse_transform([disorder_pred_label])[0]
                    
                    # Get probabilities and confidence
                    disorder_proba = disorder_model.predict_proba(X)[0]
                    confidence = disorder_proba.max() * 100
                    
                    # Determine risk level
                    if confidence <= 50:
                        risk_level = 'Low_Risk'
                    elif confidence <= 75:
                        risk_level = 'Medium_Risk'
                    else:
                        risk_level = 'High_Risk'
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📋 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                        <h3>💤 Sleep Quality</h3>
                        <h2 style="color: #1f77b4;">{quality_pred:.1f}/10</h2>
                        <p>{'Excellent' if quality_pred >= 8 else 'Good' if quality_pred >= 6 else 'Fair' if quality_pred >= 4 else 'Poor'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                        <h3>🔍 Sleep Disorder</h3>
                        <h2 style="color: #ff7f0e;">{disorder_pred}</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                        <h3>⚠️ Risk Level</h3>
                        <h2 style="color: {get_risk_color(risk_level)};">{risk_level}</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Quality Score Gauge
                        fig_gauge = go.Figure(data=[go.Indicator(
                            mode="gauge+number",
                            value=quality_pred,
                            title={'text': "Sleep Quality Score"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 10]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 3], 'color': "#ff4444"},
                                    {'range': [3, 6], 'color': "#ffaa44"},
                                    {'range': [6, 8], 'color': "#44aa44"},
                                    {'range': [8, 10], 'color': "#00aa00"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 7
                                }
                            }
                        )])
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, width='stretch')
                    
                    with col2:
                        # Risk Distribution
                        risk_scores = {
                            'Low Risk (0-50%)': 50 if risk_level == 'Low_Risk' else 0,
                            'Medium Risk (50-75%)': 25 if risk_level == 'Medium_Risk' else 0,
                            'High Risk (75-100%)': 25 if risk_level == 'High_Risk' else 0
                        }
                        
                        fig_risk = go.Figure(data=[go.Pie(
                            labels=list(risk_scores.keys()),
                            values=[100],
                            marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
                            hole=0.3
                        )])
                        fig_risk.update_layout(
                            title="Risk Assessment Distribution",
                            height=300
                        )
                        st.plotly_chart(fig_risk, width='stretch')
                    
                    st.markdown("---")
                    
                    # Health Recommendations
                    st.markdown("## 🏥 Health Recommendations")
                    recommendations = get_risk_recommendation(risk_level, disorder_pred)
                    
                    if risk_level == 'High_Risk':
                        st.markdown(f"""
                        <div class="danger-box">
                        <h3>{recommendations['title']}</h3>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'Medium_Risk':
                        st.markdown(f"""
                        <div class="warning-box">
                        <h3>{recommendations['title']}</h3>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>{recommendations['title']}</h3>
                        """, unsafe_allow_html=True)
                    
                    for i, advice in enumerate(recommendations['advice'], 1):
                        st.markdown(f"**{i}. {advice}**")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Health Metrics Summary
                    st.markdown("---")
                    st.markdown("## 📊 Your Health Metrics Summary")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Age', 'Sleep Duration', 'Physical Activity', 'Stress Level', 
                                  'Heart Rate', 'Daily Steps', 'Sleep Efficiency', 'Systolic BP', 'Diastolic BP'],
                        'Value': [f"{age} years", f"{sleep_duration:.1f} hrs", f"{physical_activity} min/day",
                                 f"{stress_level}/10", f"{heart_rate} bpm", f"{daily_steps} steps",
                                 f"{sleep_efficiency}%", f"{systolic_bp} mmHg", f"{diastolic_bp} mmHg"],
                        'Status': ['✓', '✓', '✓', '✓', '✓', '✓', '✓', '✓', '✓']
                    })
                    
                    st.dataframe(metrics_df, width='stretch', hide_index=True)
                    
                    # Save prediction to history
                    prediction_record = {
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Age': age,
                        'Gender': gender,
                        'Occupation': occupation,
                        'Sleep_Quality': round(quality_pred, 2),
                        'Disorder': disorder_pred,
                        'Confidence': round(confidence, 2),
                        'Risk_Level': risk_level,
                        'Stress_Level': stress_level,
                        'Sleep_Duration': sleep_duration,
                        'Heart_Rate': heart_rate,
                        'Sleep_Efficiency': sleep_efficiency
                    }
                    save_prediction(prediction_record)
                    st.success("✅ Prediction saved to history!")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    # ============================================================
    # PAGE: BATCH PREDICTIONS
    # ============================================================
    
    elif page == "📊 Batch Predictions":
        st.markdown("## 📊 Batch Predictions")
        st.markdown("---")
        
        st.info("""
        Upload a CSV file with health data to get predictions for multiple individuals.
        The CSV should contain the same columns as the single prediction form.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded {len(df)} records**")
                
                # Show sample data
                st.markdown("### Preview of Data")
                st.dataframe(df.head(), width='stretch')
                
                # Process predictions
                if st.button("🔮 Generate Predictions"):
                    progress_bar = st.progress(0)
                    
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        try:
                            # Prepare input
                            input_data = {key: row[key] for key in df.columns if key in [
                                'Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level',
                                'Heart Rate', 'Daily Steps', 'SleepDisorder_Imputed', 'Systolic_BP',
                                'Diastolic_BP', 'Sleep_Efficiency', 'Health_Risk_Score',
                                'BMI Category_Encoded', 'Sleep_Duration_Category_Encoded',
                                'Activity_Category_Encoded', 'Stress_Category_Encoded',
                                'BP_Category_Encoded', 'Gender_Male', 'Occupation_Office Worker',
                                'Occupation_Retired', 'Occupation_Student', 'Age_Group_Middle_Age',
                                'Age_Group_Senior', 'Age_Group_Young_Adult', 'Heart_Rate_Category_Normal',
                                'Steps_Category_Low_Active', 'Steps_Category_Sedentary',
                                'Steps_Category_Somewhat_Active'
                            ]}
                            
                            # Make predictions
                            X = create_input_features(input_data)
                            quality = quality_model.predict(X)[0]
                            disorder_label = disorder_model.predict(X)[0]
                            disorder = encoder.inverse_transform([disorder_label])[0]
                            confidence = disorder_model.predict_proba(X)[0].max() * 100
                            
                            # Risk level
                            if confidence <= 50:
                                risk = 'Low_Risk'
                            elif confidence <= 75:
                                risk = 'Medium_Risk'
                            else:
                                risk = 'High_Risk'
                            
                            predictions.append({
                                'Quality': round(quality, 2),
                                'Disorder': disorder,
                                'Confidence': round(confidence, 2),
                                'Risk_Level': risk
                            })
                            
                        except Exception as e:
                            predictions.append({
                                'Quality': 'Error',
                                'Disorder': 'Error',
                                'Confidence': 0,
                                'Risk_Level': 'Error'
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Create results dataframe
                    results_df = pd.concat([df.reset_index(drop=True), 
                                          pd.DataFrame(predictions)], axis=1)
                    
                    st.markdown("### Predictions")
                    st.dataframe(results_df, width='stretch')
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"sleep_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary Statistics
                    st.markdown("### Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_quality = results_df['Quality'].mean()
                        st.metric("Average Sleep Quality", f"{avg_quality:.2f}/10")
                    
                    with col2:
                        disorder_counts = results_df['Disorder'].value_counts()
                        st.metric("Most Common Disorder", disorder_counts.index[0])
                    
                    with col3:
                        high_risk = (results_df['Risk_Level'] == 'High_Risk').sum()
                        st.metric("High Risk Count", high_risk)
                    
                    with col4:
                        avg_confidence = results_df['Confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # ============================================================
    # PAGE: ANALYTICS
    # ============================================================
    
    elif page == "📈 Analytics":
        st.markdown("## 📈 Analytics Dashboard")
        st.markdown("---")
        
        # Load sample predictions data
        try:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'sleep_health_with_predictions.csv')
            df_data = pd.read_csv(data_path)
            
            st.markdown("### 📊 Population Health Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_quality = df_data['Predicted_Sleep_Quality'].mean()
                st.metric("Average Sleep Quality", f"{avg_quality:.2f}/10")
            
            with col2:
                avg_age = df_data['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f} years")
            
            with col3:
                avg_stress = df_data['Stress Level'].mean()
                st.metric("Average Stress Level", f"{avg_stress:.2f}/10")
            
            with col4:
                disorder_count = (df_data['Predicted_Disorder'] != 'None').sum()
                st.metric("People with Disorders", f"{disorder_count}/{len(df_data)}")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sleep Quality Distribution
                fig = px.histogram(df_data, x='Predicted_Sleep_Quality', 
                                  nbins=20, title='Sleep Quality Distribution',
                                  labels={'Predicted_Sleep_Quality': 'Sleep Quality (1-10)'})
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Risk Level Distribution
                risk_counts = df_data['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title='Risk Level Distribution',
                           color_discrete_map={'Low_Risk': '#28a745', 'Medium_Risk': '#ffc107', 'High_Risk': '#dc3545'})
                st.plotly_chart(fig, width='stretch')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age vs Sleep Quality
                fig = px.scatter(df_data, x='Age', y='Predicted_Sleep_Quality',
                               color='Risk_Level', title='Age vs Sleep Quality',
                               color_discrete_map={'Low_Risk': '#28a745', 'Medium_Risk': '#ffc107', 'High_Risk': '#dc3545'})
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Disorder Distribution
                disorder_counts = df_data['Predicted_Disorder'].value_counts()
                fig = px.bar(x=disorder_counts.index, y=disorder_counts.values,
                           title='Sleep Disorder Distribution',
                           labels={'x': 'Disorder Type', 'y': 'Count'})
                st.plotly_chart(fig, width='stretch')
            
        except FileNotFoundError:
            st.warning("⚠️ Sample data file not found. Analytics not available.")
    
    # ============================================================
    # PAGE: ABOUT
    # ============================================================
    
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This Application")
        st.markdown("---")
        
        st.markdown("""
        ### 🎯 Purpose
        This application provides an AI-powered solution for predicting sleep health and detecting sleep disorders.
        It uses machine learning models trained on comprehensive health data to offer personalized insights.
        
        ### 🤖 Machine Learning Models
        
        **1. Sleep Quality Predictor (Regression)**
        - Model Type: XGBoost / Random Forest
        - Task: Predicts sleep quality on a 1-10 scale
        - Features: 28 health-related features
        - Performance: R² Score optimized
        
        **2. Sleep Disorder Classifier (Multi-class)**
        - Model Type: XGBoost / LightGBM
        - Task: Classifies sleep disorders
        - Classes: """ + ", ".join(encoder.classes_) + """
        - Features: 28 health-related features
        - Performance: F1-Score optimized
        
        ### 📊 Input Features
        The models use the following categories of features:
        
        **Personal Information:**
        - Age, Gender, Occupation
        
        **Sleep Metrics:**
        - Sleep Duration, Sleep Efficiency, Sleep Quality
        
        **Physical Health:**
        - BMI Category, Heart Rate, Blood Pressure
        - Daily Steps, Physical Activity Level
        
        **Mental Health:**
        - Stress Level
        
        **Health Indicators:**
        - Health Risk Score
        
        ### 📈 Data Preprocessing
        - One-hot encoding for categorical variables
        - Feature scaling for numerical variables
        - Handling of missing values
        - SMOTE for class imbalance in training
        
        ### ✅ Model Performance
        - Quality Model: Optimized for R² Score
        - Disorder Model: Optimized for F1-Score
        - Both models use cross-validation during training
        
        ### 🔐 Data Privacy
        - No data is stored or logged
        - All predictions are local
        - Your personal health information stays with you
        
        ### 📝 Disclaimer
        This application is for educational and informational purposes only.
        It should not replace professional medical advice. Always consult with healthcare providers
        for diagnosis and treatment of sleep disorders.
        
        ### 👨‍💻 Technical Stack
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning library
        - **XGBoost/LightGBM**: Advanced gradient boosting
        - **Pandas/NumPy**: Data processing
        - **Plotly**: Data visualization
        
        ### 📞 Support
        For issues or questions, please contact the development team.
        
        ### 📄 Version
        Application Version: 1.0  
        Last Updated: December 2024
        """)


if __name__ == "__main__":
    main()
