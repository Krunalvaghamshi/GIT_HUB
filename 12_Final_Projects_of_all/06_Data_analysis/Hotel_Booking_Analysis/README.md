**Project Title:** Hotel Booking Cancellation Analysis & Prediction

**Problem Statement:**
The hotel industry faces significant challenges with booking cancellations, which lead to revenue loss and operational inefficiencies. This project aims to analyze hotel booking data to understand the factors influencing cancellations and build predictive models to forecast whether a booking will be canceled. The goal is to provide actionable insights for hotel management to reduce cancellations and optimize revenue.

### **Dataset Description**

The dataset contains booking information for a **City Hotel** and a **Resort Hotel**, including various attributes related to the booking, customer, and stay.
* **Source:** The dataset originates from a Kaggle competition and was originally detailed in the article "Hotel Booking Demand Datasets" by Nuno Antonio et al..
* **Size:** The dataset contains approximately 119,390 rows and 32 columns.
* **Key Features:**
    * **Booking Information:** `is_canceled` (Target Variable), `lead_time`, `arrival_date_year/month/day`, `stays_in_weekend_nights`, `stays_in_week_nights`, `booking_changes`, `deposit_type`, `days_in_waiting_list`, `adr` (Average Daily Rate), `total_of_special_requests`.
    * **Customer Demographics:** `adults`, `children`, `babies`, `meal`, `country`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `customer_type`.
    * **Historical Data:** `previous_cancellations`, `previous_bookings_not_canceled`.
    * **Room Details:** `reserved_room_type`, `assigned_room_type`.

### **Development Journey**

The project followed a structured data science pipeline, divided into several key phases:

#### **Phase 1: Data Cleaning & Preprocessing**
The raw data required significant cleaning to ensure quality and reliability for analysis.
1.  **Handling Missing Values:**
    * The dataset contained null values in columns like `children`, `country`, `agent`, and `company`.
    * Missing values were filled with 0 for numerical columns where appropriate, or handled based on context (e.g., assuming no agent was involved if `agent` is null).
2.  **Data Filtering:**
    * Rows where the number of `adults`, `children`, and `babies` were all zero were identified as invalid bookings and removed from the dataset to ensure data integrity.
3.  **Feature Engineering:**
    * New features were derived, such as converting `reservation_status_date` to datetime objects and extracting `year`, `month`, and `day`.
    * Total guests were calculated by summing adults, children, and babies.

#### **Phase 2: Exploratory Data Analysis (EDA)**
Extensive EDA was performed to uncover patterns and trends.
1.  **Cancellation Analysis:**
    * The overall cancellation rate was visualized. It was observed that City Hotels generally have a higher number of bookings and cancellations compared to Resort Hotels.
    * 2.  **Seasonality & Trends:**
    * Booking volumes and prices (`adr`) vary significantly throughout the year. August is typically the busiest month, while prices fluctuate based on demand and seasonality.
    * 3.  **Customer Behavior:**
    * The analysis looked into market segments (e.g., Online TA, Corporate) to see which channels bring the most bookings and which have higher cancellation rates.
    * The impact of lead time on cancellations was investigated, revealing that bookings made far in advance are more likely to be canceled.
4.  **Geographical Analysis:**
    * Bookings were analyzed by country to identify the top source markets for the hotels.
5.  **Room Type Analysis:**
    * The relationship between reserved vs. assigned room types was explored. Changes in room assignment were checked for their correlation with cancellations.

#### **Phase 3: Model Building & Prediction**
To predict cancellations, machine learning models were developed.
1.  **Data Preparation for Modeling:**
    * Categorical variables were encoded (e.g., using One-Hot Encoding or Label Encoding) to convert them into a numerical format suitable for algorithms.
    * The dataset was split into training and testing sets.
2.  **Model Selection:**
    * Various classification algorithms were likely tested, such as Logistic Regression, Decision Trees, or Random Forests (as implied by the `sklearn` imports in the notebook).
3.  **Model Evaluation:**
    * Models were evaluated using metrics like Accuracy, Confusion Matrix, and Classification Report to assess their performance in correctly identifying cancellations.
    * Features like `lead_time`, `market_segment`, and `deposit_type` were identified as "Smart Drivers" or key predictors for the model.

### **Key Insights & Conclusion**
* **High Cancellation Rate:** A significant portion of bookings is canceled, impacting revenue management.
* **Lead Time Impact:** Longer lead times are strongly correlated with higher cancellation probabilities.
* **Deposit Policies:** The type of deposit (No Deposit, Non-Refundable, Refundable) plays a crucial role in cancellation behavior.
* **Seasonality:** Revenue and occupancy are highly seasonal, with peaks in summer months.
* **Market Segments:** Online Travel Agents (Online TA) are a major source of bookings but may also have distinct cancellation patterns compared to direct bookings or corporate clients.

### **Future Scope**
* **Dynamic Pricing:** Implementing dynamic pricing strategies based on cancellation probabilities to maximize revenue.
* **Targeted Marketing:** Creating personalized offers for customers with a lower likelihood of cancellation.
* **Real-time Prediction:** Deploying the model into a real-time system to flag high-risk bookings as they are made.

---

### **Links**

* **GitHub Repo:** https://github.com/Krunalvaghamshi/GIT_HUB/tree/7e9c1307bad179b2d433ee70e1c298f56726febd/12_Final_Projects_of_all/06_Data_analysis/Hotel_Booking_Analysis

* **Portfolio:** https://kruvs-portfolio.vercel.app/

* **Documentation:** https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/documentation_of_hotel_booking_analysis_and_further.html

