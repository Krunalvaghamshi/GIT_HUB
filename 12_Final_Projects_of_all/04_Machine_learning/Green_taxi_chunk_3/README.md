# **Project: NYC Smart Taxi Predictor**

### **Project Overview**
The **NYC Smart Taxi Predictor** is a sophisticated machine learning application designed to estimate taxi fares in New York City with high precision. By analyzing historical trip data from Green Taxis, this system provides users with real-time fare predictions based on pickup/dropoff locations, time of day, and passenger count. The project bridges the gap between raw data analysis and a user-friendly, interactive web interface, serving as a comprehensive tool for both passengers and drivers to plan trips effectively.

---

### **Development Journey**

The development of this project followed a structured data science lifecycle, evolving from raw data exploration to a deployed intelligent web application.

#### **Phase 1: Data Discovery & Preprocessing**
* **Data Source:** The project utilizes the **2013 Green Taxi Dataset** (specifically "Chunk 3"), which contains detailed logs of trips including GPS coordinates, timestamps, and fare breakdowns.
* **Data Cleaning:**
    * Handled missing values and cryptic IDs (e.g., mapping `RateCodeID: 2` to "JFK Airport").
    * **Outlier Removal:** Filtered out unrealistic data points to ensure model stability. This included removing trips with zero distance, negative fares, or fares exceeding $100, as well as limiting trip distances to under 50 miles.
    * **Formatting:** Converted `pickup_datetime` and `dropoff_datetime` strings into proper datetime objects for temporal analysis.

#### **Phase 2: Feature Engineering**
To improve the model's ability to learn patterns, several new features were derived from the raw data:
* **Trip Duration:** Calculated the difference between drop-off and pick-up times in minutes.
* **Temporal Features:** Extracted critical components like `pickup_hour`, `pickup_day_of_week`, and a boolean flag `is_weekend` to capture traffic patterns associated with specific times and days.
* **Distance Transformation:** Applied a logarithmic transformation (`log_trip_distance`) to the trip distance variable to handle the right-skewed distribution of trip lengths (normalizing short vs. long trips).
* **Interaction Terms:** Created features like `distance_duration_interaction` to capture the relationship between how far a passenger travels and how long it takes.

#### **Phase 3: Exploratory Data Analysis (EDA)**
Extensive EDA was performed using libraries like **Seaborn** and **Matplotlib** to understand the underlying distributions:
* **Univariate Analysis:** Analyzed the distribution of passenger counts (revealing most trips are single-passenger) and tip amounts (showing a high frequency of $0 tips, likely cash payments).
* **Bivariate Analysis:** Investigated correlations, such as the positive relationship between **Trip Distance and Fare Amount**, and the impact of **Payment Type** on total costs.
* **Profiling:** Utilized advanced profiling tools to generate comprehensive reports on data health and correlations.

#### **Phase 4: Model Development & Training**
* **Algorithm Selection:** **XGBoost Regressor** was chosen for its efficiency and high performance with structured tabular data.
* **Training:** The model was trained on a sample of **100,000 records** to balance performance and speed.
* **Performance:** The final model achieved an impressive **R² Score of 0.97**, indicating it can explain 97% of the variance in fare prices.
* **Artifacts:** The best-performing model was serialized and saved as `best_model_green_taxi_trained_on_100000_sample.pkl` for real-time inference in the app.

#### **Phase 5: Application Development (Streamlit)**
The final phase involved building an interactive frontend using **Streamlit**:
* **UI/UX Design:** Implemented a **"Premium Dark Mode"** with neon green accents (`#00CC96`) and glassmorphism cards to give the app a modern, futuristic feel.
* **Geo-Intelligence:** Integrated **PyDeck** for 3D map visualizations. Users can select from over **40+ NYC landmarks** (e.g., Times Square, JFK Airport) or input custom coordinates.
* **Smart Logic:**
    * **Traffic Simulation:** The app calculates distance using the Haversine formula with a **1.3x City Traffic Multiplier** to estimate realistic travel paths rather than straight lines.
    * **Dynamic Pricing:** It automatically detects and applies surcharges for **Rush Hour** (4 PM - 8 PM) and **Overnight** travel (8 PM - 6 AM).

---

### **System Specifications & Tech Stack**

* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib, PyDeck
* **Machine Learning:** Scikit-Learn, XGBoost
* **Web Framework:** Streamlit
* **Serialization:** Joblib

### **Key Features**
1.  **Real-Time Predictions:** Instant fare estimates based on live user inputs.
2.  **Smart Insights:** The app provides context, such as identifying "Heavy Traffic" or "Clear Roads" based on the time of day.
3.  **Visual Analytics:** Interactive charts and 3D maps allow users to verify their route visually.
4.  **Surcharge Detection:** Automatic calculation of NYC taxi surcharges for accurate pricing.

---

### **Project Links**

* **GitHub Repository:** https://github.com/Krunalvaghamshi/GIT_HUB/tree/7e9c1307bad179b2d433ee70e1c298f56726febd/12_Final_Projects_of_all/04_Machine_learning/Green_taxi_chunk_3

* **Project Documentation:** https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/documentation_green_taxi_app_v1.html

* **Developer Portfolio:** https://kruvs-portfolio.vercel.app/

* **Streamlit Application:** https://app-rcrknxzyxrit5emlknw4wm.streamlit.app/


