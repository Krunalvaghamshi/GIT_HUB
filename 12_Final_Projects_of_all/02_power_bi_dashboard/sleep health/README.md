# Project Name: Strategic Sleep Health & Risk Analysis (SleepAI Analytics)

### **1. Executive Summary**
This is a comprehensive, end-to-end Data Science and Business Intelligence project designed to transform raw lifestyle data into actionable health intelligence. Moving beyond static reporting, this system utilizes a dataset of 400+ individuals to identify the root causes of sleep disorders.

The project ecosystem consists of three integrated components:
1.  **Machine Learning Engine:** Predictive models achieving **93.75% classification accuracy** and **98% regression accuracy (R²)**.
2.  **Strategic Dashboard (Power BI):** A 3-page interactive interface for population health management and risk segmentation.
3.  **Diagnostic Application (Streamlit):** A user-facing, clinical-grade AI tool for individual risk assessment.

---

### **2. Business Problem & Objectives**
Sleep health is often treated as an isolated issue. The objective of this project was to prove that poor sleep is actually a **symptom** of broader lifestyle factors. The goal was to provide stakeholders with a data-driven "call to action" to pivot wellness programs from general sleep hygiene to targeted stress management and obesity intervention.

**Key Performance Indicators (KPIs):**
* **Disorder Rate:** 27.5% of the analyzed population.
* **Average Sleep Quality:** 6.13 / 10.
* **Correlation Findings:** Established a strong negative correlation (-0.81) between Stress Levels and Sleep Quality.

---

### **3. Development Journey (The Lifecycle)**

This project followed a rigorous 5-phase development lifecycle:

#### **Phase 1: Data Exploration (EDA) & Ingestion**
* **Ingestion:** Loaded a dataset of 400 unique records covering sleep metrics and lifestyle habits.
* **Data Cleaning:** Addressed complex data structures, such as splitting string-based Blood Pressure values (e.g., "126/83") into usable numerical features (`Systolic` and `Diastolic`).
* **Initial Discovery:** Identified a massive class imbalance in sleep disorders, signaling the need for synthetic data generation later in the pipeline.

#### **Phase 2: Advanced Feature Engineering**
To improve model performance, raw data was enriched with derived metrics:
* **`Sleep_Efficiency`:** Calculated a normalized metric comparing time spent in bed vs. actual sleep duration.
* **`Health_Risk_Score`:** Engineered a weighted composite score combining BMI, stress, and blood pressure to quantify holistic health risk.
* **Encoding Strategies:** Applied One-Hot Encoding for nominal data (Gender) and Ordinal Encoding for ranked data.
* **Zero-Vector Fix:** Developed a specific logic to handle feature mismatches during single-row prediction scenarios in the deployment phase.

#### **Phase 3: Machine Learning Model Development**
The "Brain" of the system was developed using Python (Scikit-learn/XGBoost):
* **Classification (Disorder Prediction):** Trained a **Random Forest Classifier** to predict the presence of Insomnia or Sleep Apnea.
    * *Optimization:* Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset, resulting in a **93.75% Accuracy** and high F1-score.
* **Regression (Quality Prediction):** Trained an **XGBoost Regressor** to predict specific Sleep Quality scores (1-10).
    * *Optimization:* Achieved an **R² of ~0.98**, explaining nearly all variance in the data.
* **Serialization:** Models were serialized into `.pkl` files for lightweight deployment.

#### **Phase 4: Strategic Analytics (Power BI)**
Designed a "Dark Mode" Glassmorphism dashboard to visualize the findings:
* **Page 1 (Executive Summary):** Displays top-level KPIs. Features a critical scatter plot proving that as Stress Levels exceed 5/10, Sleep Quality plummets and disorders appear.
* **Page 2 (Demographic Insights):** Segments data by Occupation and BMI. Discovered "Latent Risk" in Office Workers—who have high stress but average reported sleep quality—marking them as a vulnerable group. Confirmed the link between Obesity and Sleep Apnea.
* **Page 3 (Risk Analysis):** The "Actionable Plan." This page utilizes the ML predictions to rank individuals by `Disorder_Confidence` (diagnostic) and `Health_Risk_Score` (prognostic), providing a priority list for intervention.

#### **Phase 5: Intelligent Application (Streamlit)**
Built a user-facing web application for real-time diagnosis:
* **Smart Logic:** Includes a context-aware recommendation engine that suggests specific lifestyle changes based on input.
* **Persistence:** Implemented a hybrid Session+CSV storage system to prevent data loss during cloud, multi-threaded usage.
* **UI/UX:** Designed with a premium dark interface to match the Power BI aesthetic.

---

### **4. Key Analytical Insights**
Based on the final report, the project delivered three major strategic conclusions:
1.  **Stress is the #1 Driver:** High stress is the single most powerful predictor of poor sleep. Wellness programs must pivot to stress management.
2.  **Hidden Risk in Office Workers:** While Engineers report good sleep, Office Workers and Sales Representatives suffer from high stress and "latent" risk, requiring preventative care.
3.  **Age-Based Variances:** Seniors get *more* sleep (duration) but *worse* sleep (quality/fragmentation), whereas Young Adults sleep less but more efficiently. Interventions must be age-specific.

---

### **Project Links**

* **GitHub Repository:** https://github.com/Krunalvaghamshi/GIT_HUB/tree/7af962d02fadd16e8aa1a680f31e42e41c9d892e/12_Final_Projects_of_all/02_power_bi_dashboard/sleep%20health
* **Project Documentation:** https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/Documentaion_sleep_health_dashboard.html
* **Portfolio link:** https://kruvs-portfolio.vercel.app/
