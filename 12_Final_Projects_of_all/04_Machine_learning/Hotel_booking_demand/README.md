# 🏨 Project Name: Hotel IQ | Revenue Concierge

### **Project Overview**
**Hotel IQ** is an end-to-end Machine Learning application designed to solve one of the hospitality industry's most critical challenges: **Booking Cancellations**.

By leveraging historical booking data, this tool predicts the *exact probability* of a customer canceling their reservation. However, it goes beyond simple prediction. It acts as a **"Revenue Concierge"** by offering dynamic simulations—allowing hotel managers to adjust variables like **Average Daily Rate (ADR)** or **Lead Time** to see how these changes reduce cancellation risk in real-time.

---

### **📊 The Data Landscape**
The project is built upon a robust dataset containing approximately **119,390 observations** representing booking data for a City Hotel and a Resort Hotel.
* **Source:** Hotel Booking Demand Dataset (originally from Nuno Antonio et al.)
* **Key Features Analyzed:**
    * **Booking Logistics:** `lead_time`, `arrival_date_week_number`, `stays_in_weekend_nights`, `stays_in_week_nights`.
    * **Customer Demographics:** `adults`, `children`, `babies`, `country`, `market_segment`.
    * **Financials:** `adr` (Average Daily Rate), `deposit_type`.
    * **History:** `is_repeated_guest`, `previous_cancellations`.
    * **Target Variable:** `is_canceled` (0 = Checked-in, 1 = Canceled).

---

### **🚀 The Development Journey**

#### **Phase 1: Exploratory Data Analysis (EDA) & Data Cleaning**
Before modeling, a deep dive into the data was conducted using **Pandas**, **Matplotlib**, and **Seaborn** (evident in your `classification_algorithms...ipynb` notebook).
* **Null Handling:** You identified significant missing values in the `company` (94% missing) and `agent` (13% missing) columns. These were handled strategically (e.g., filling with 0 to indicate direct bookings) rather than dropping valuable data.
* **Pattern Recognition:** You visualized correlations using heatmaps to understand how features like `lead_time` (time between booking and arrival) and `deposit_type` drastically affect cancellation rates.
* **Data Encoding:** Categorical variables like `market_segment` and `distribution_channel` were encoded (via Label/One-Hot encoding) to make them machine-readable.

#### **Phase 2: Model Selection & Training**
The core intelligence of Hotel IQ is powered by the **XGBoost Classifier** (`xgboost_booking_model.pkl`).
* **Algorithm Choice:** XGBoost was selected for its efficiency with tabular data and ability to handle the complex, non-linear relationships between booking details and cancellation behavior.
* **Pipeline:** The project uses a `joblib` pipeline that bundles the trained model with a `StandardScaler`. This ensures that any new input data from the web app is scaled exactly like the training data before prediction, preventing data leakage.

#### **Phase 3: Application Architecture (Streamlit)**
The interface is built using **Streamlit**, customized with a "Luxury Hotel Theme" using custom CSS injection (Playfair Display fonts, Gold/Midnight Blue color palette).
* **Input Layer:** A user-friendly sidebar allows hotel staff to input guest details (Dates, Adults, Meal Plan, Market Segment, etc.).
* **The Logic Core:**
    * The app calculates derived features like `total_nights` automatically from check-in/out dates.
    * It fetches the probability of cancellation (`model.predict_proba`) rather than just a binary Yes/No.
* **Visualization:** It uses **Plotly Graph Objects** to render a custom **Risk Gauge**, providing immediate visual feedback on the safety of the booking.

#### **Phase 4: The Revenue Concierge (Simulation Engine)**
This is the standout feature of your project (found in `streamlit_app_hotel_booking_demand.py`).
* **What-If Analysis:** If a booking has a high cancellation risk (e.g., 75%), the app allows the user to simulate changes:
    * *What if we lower the price (ADR)?*
    * *What if they book closer to the date (Lead Time)?*
* **Dynamic Feedback:** As the user moves the sliders, the app re-runs the XGBoost prediction in real-time, showing the **"NEW RISK"** and the **"RISK REDUCTION"** percentage. This empowers managers to make data-driven offers to save a booking.

---

### **🛠️ Tech Stack**
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn, Joblib
* **Visualization:** Plotly, Matplotlib, Missingno
* **Web Framework:** Streamlit (with custom HTML/CSS injection)
* **Deployment:** Streamlit Cloud

---

### **🔗 Project Links**

* **GitHub Repository:** https://github.com/Krunalvaghamshi/GIT_HUB/tree/7e9c1307bad179b2d433ee70e1c298f56726febd/12_Final_Projects_of_all/04_Machine_learning/Hotel_booking_demand

* **Live Application:** https://app-a8dfjh6cabrzrkbrngqpng.streamlit.app/

* **Project Documentation:** https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/documentation_of_hotel_booking_.html

* **Developer Portfolio:** https://kruvs-portfolio.vercel.app/
