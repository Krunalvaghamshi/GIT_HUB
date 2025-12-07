### **Project Title: Telecom Churn Analysis Dashboard**

#### **Project Overview**
This project is a comprehensive data analytics solution designed to analyze customer churn within the telecom sector. It focuses on identifying patterns in customer behavior, usage habits, and demographic factors that lead to service cancellation. By visualizing key metrics across major telecom partners (Reliance Jio, Airtel, Vodafone, and BSNL), the dashboard empowers stakeholders to make data-driven decisions to improve customer retention.

The analysis is built upon a dataset of **telecom records**, providing a granular view of customer interactions, from call duration and data usage to customer service calls and plan types.

---

### **Development Journey**

The development of this project followed a structured lifecycle, ensuring data accuracy and insightful reporting:

#### **1. Data Acquisition & Understanding**
* **Source Data:** The project started with raw customer data (`telecom_churn.csv`), which includes critical fields such as `customer_id`, `telecom_partner`, `gender`, `age`, `state`, `city`, `pincode`, `date_of_registration`, `estimated_salary`, and usage metrics like `calls_made`, `sms_sent`, and `data_used`.
* **Initial Audit:** A preliminary check was likely performed to understand the distribution of users across different states and partners.

#### **2. Data Cleaning & Transformation**
* **Data Processing:** The development involved cleaning "dirty data" to ensure accuracy. This step likely handled missing values, corrected data types, and standardized text fields to prepare the dataset for analysis.
* **Feature Engineering:** New columns and derived metrics were likely created to facilitate deeper analysis, evidenced by the `telecom_churn_expiremental.xlsx` file which suggests an iterative process of refining the data structure for better reporting.

#### **3. Data Modeling & DAX Implementation**
A robust analytical layer was built using **DAX (Data Analysis Expressions)** to calculate dynamic metrics. This allowed for real-time aggregation based on user filters. Key measures created include:
* **Customer Metrics:** `Total TeleCome Users`, `Total TeleCom Users churned`, and `Total Stayed TeleCom Users` were calculated to track the base health.
* **Usage Analysis:** Complex aggregates were created for `Total Calls Made`, `Total SMS Sent`, and `Total Data Used` to measure engagement.
* **Provider-Specific Analysis:** Specific measures were written for each partner (e.g., `Jio Churned Customer`, `Airtel Total Customers`, `Vodafone Stayed Customers`) to allow for side-by-side comparison.
* **Advanced Analytics:** You implemented logic to find the "Most Frequent Age" of users per carrier (e.g., `MostFrequentAgeBSNL`), which helps in targeted marketing.

#### **4. Dashboard Design & Visualization**
The final output is a multi-page Power BI report (`tele_com_dashboard.pbix`) that visualizes the insights:
* **Executive Summary:** A high-level view showing total users (244K), retention rates, and churn split by company. It highlights that retention is relatively consistent across providers (~25%).
* **Demographic Insights:** Visuals comparing churn and usage by Gender and Age. For instance, you identified specific age groups with higher churn risk.
* **Usage Trends:** Analysis of "Data Used," "SMS Sent," and "Calls Made" over time (Month Wise) to identify seasonal trends or service quality issues.
* **Geographic Distribution:** Maps showing user density and churn hotspots across India, helping to pinpoint regional service issues.

---

### **Key Insights Discovered**
* **User Demographics:** There is a distinct age profile for different carriers. For example, **Vodafone has an aging user base (Mode Age: 73)**, whereas competitors have a younger demographic (avg age ~49-51).
* **Churn Rate Consistency:** The churn rate is alarmingly consistent across all major providers (Reliance Jio, Airtel, Vodafone, BSNL), hovering around the 25% mark. This suggests a highly competitive market where users switch frequently.
* **High Value Customers:** You have successfully identified high-value segments by analyzing "Estimated Salary" against "Data Used," allowing for premium plan targeting.

---

### **Project Links**

* **GitHub Repository:** https://github.com/Krunalvaghamshi/GIT_HUB/tree/3940aed3b5af12da16f876e85332728c5216eed3/12_Final_Projects_of_all/02_power_bi_dashboard/telecom%20company%20churn
* **Live Portfolio:** https://kruvs-portfolio.vercel.app/
* **Project Documentation:** https://raw.githack.com/Krunalvaghamshi/GIT_HUB/2845f8a66d7b8dd775926b27c07a979001f78d99/12_Final_Projects_of_all/00_Documentation/Documentation_telecom_churn.html