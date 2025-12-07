-----

# 🌙 Project Name: Sleep Health & Lifestyle Intelligence System

### 📖 **Project Overview**

This project is an end-to-end data science solution designed to analyze human sleep patterns, lifestyle habits, and cardiovascular metrics to predict sleep health outcomes. It utilizes advanced Machine Learning algorithms to perform two key tasks:

1.  **Regression:** Predicting a continuous **Sleep Quality Score (1-10)**.
2.  **Classification:** Diagnosing potential **Sleep Disorders** (Insomnia, Sleep Apnea, or Healthy).

The system culminates in a user-friendly **Streamlit Web Application** that acts as a "Virtual Sleep Doctor," providing real-time diagnostics and personalized health recommendations based on user bio-markers.

-----

### 🚀 **Development Journey**

The project execution followed the industry-standard **CRISP-DM** (Cross-Industry Standard Process for Data Mining) lifecycle.

#### **Phase 1: Exploratory Data Analysis (EDA)**

  * **Objective:** To understand the underlying structure of the "Sleep Health and Lifestyle Dataset."
  * **Key Actions:**
      * Loaded raw data containing demographics, sleep metrics, and daily habits.
      * Standardized column names (e.g., converting "Physical Activity Level (minutes/day)" to "Physical Activity Level") for consistency.
      * **Visualization:** Created histograms and count plots to visualize distributions of Sleep Duration, Quality, and BMI.
      * **Discovery:** Identified a significant class imbalance in Sleep Disorders (Healthy vs. Disorders) and a strong correlation between Stress Levels and Sleep Quality.

#### **Phase 2: Data Preprocessing & Feature Engineering**

  * **Objective:** To transform raw data into a format suitable for high-performance ML models.
  * **Key Actions:**
      * **Missing Value Imputation:** Handled missing values in the "Sleep Disorder" column by creating a "None" category for healthy individuals.
      * **Feature Extraction:**
          * *Blood Pressure:* Split into `Systolic_BP` and `Diastolic_BP` and categorized into medical ranges (Normal, Elevated, Hypertension).
      * **Feature Creation (The "Expert Twist"):**
          * `Sleep_Efficiency`: Calculated as `(Sleep Duration / 8) * Sleep Quality`.
          * `Health_Risk_Score`: A composite weighted score derived from BMI, Stress, Heart Rate, and Activity levels.
      * **Encoding:** Applied One-Hot Encoding and Label Encoding to categorical variables (Gender, Occupation, BMI Category).
      * **Scaling:** Used `StandardScaler` to normalize numerical features for model stability.

#### **Phase 3: Advanced Analysis**

  * **Objective:** To uncover deeper relationships between lifestyle choices and sleep health.
  * **Insights Generated:**
      * **Occupation Impact:** Identified that *Sales Representatives* and *Scientists* often reported lower sleep quality compared to *Engineers*.
      * **Stress-Sleep Correlation:** Confirmed a strong negative correlation (-0.89) between stress levels and sleep duration.
      * **BMI Factor:** Found that Obese and Overweight individuals had a significantly higher likelihood of Sleep Apnea.

#### **Phase 4: Machine Learning Modeling**

  * **Objective:** To build and tune predictive models.
  * **Approach:**
      * **Data Splitting:** 80/20 train-test split.
      * **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the Sleep Disorder classes.
      * **Model Selection:** Trained and evaluated multiple algorithms:
          * *Regression (Quality):* Linear Regression, Ridge, **Random Forest**, XGBoost.
          * *Classification (Disorder):* Logistic Regression, **Random Forest**, XGBoost, LightGBM.
      * **Result:** **Random Forest** emerged as the champion model for both tasks due to its ability to handle non-linear relationships and provide high accuracy (F1-Score \> 0.89).

#### **Phase 5: Application Deployment (Streamlit)**

  * **Evolution:** The app evolved through three major versions:
      * **v1.0 (MVP):** Basic input form and raw prediction output.
      * **v2.0 (Enhanced UI):** Introduced a dark-mode "Premium UI" with custom CSS styling and better error handling.
      * **v3.0 (Final Product):** A "Vision-Full" application featuring:
          * **Smart Recommendations:** Context-aware advice (e.g., suggesting CBT-I techniques for Insomnia).
          * **History Tracking:** Saves recent predictions to a local CSV log.
          * **Confidence Scores:** Displays how certain the AI is about its diagnosis.

-----

### 💻 **Technical Architecture**

```text
[ User Input ]  ->  [ Streamlit UI ]  ->  [ Feature Engineering Pipeline ]
                                                      |
                                                      v
                                            [ Model Loader Class ]
                                                      |
           +--------------------------+---------------+-------------------------+
           |                          |                                         |
[ Sleep Quality Model ]     [ Disorder Classifier ]                    [ Risk Calculator ]
   (Random Forest Reg)      (Random Forest Clf)                     (Python Logic)
           |                          |                                         |
           v                          v                                         v
[  Quality Score (1-10) ]   [ Diagnosis (e.g., Apnea) ]              [ Risk Level (High/Low) ]
           |                          |                                         |
           +--------------------------+---------------+-------------------------+
                                                      |
                                                      v
                                        [ Recommendation Engine ]
                                          (Generates Action Plan)
```

-----

### ✨ **Key Application Features**

1.  **Dual-Mode Prediction:** Predicts both a numerical sleep score and a categorical medical condition simultaneously.
2.  **Personalized Action Plan:** Provides tailored advice based on specific risk factors (e.g., "High Stress & Low Sleep" triggers specific relaxation protocols).
3.  **Real-Time Analytics:** Users can view their prediction history and download reports.
4.  **Interactive Dashboard:** Uses Plotly for visualizing health metrics against population averages.

-----

### 🛠️ **Tech Stack**

  * **Language:** Python 3.10+
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn, Plotly
  * **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE)
  * **Web Framework:** Streamlit
  * **Model Persistence:** Pickle

-----

### 🔗 **Project Links**

  * **GitHub Repository:** `[https://github.com/Krunalvaghamshi/GIT_HUB/tree/3940aed3b5af12da16f876e85332728c5216eed3/12_Final_Projects_of_all/01_Analysis]`
  * **Project Portfolio:** `[https://kruvs-portfolio.vercel.app/]`
  * **Documentation:** `[https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/Documentation_of_sleep_health_app.html]`
  * **Streamlit Application:** `[https://app-vuytw223yxmfjabq76neu4.streamlit.app/]`