# Sleep Health & Disorder Prediction System - Streamlit Application

## 📋 Overview

This is a comprehensive Streamlit web application for predicting sleep quality and detecting sleep disorders using machine learning models. The application provides:

- **Individual Predictions**: Get personalized sleep quality and disorder predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: View population-level health insights
- **Risk Assessment**: Personalized health recommendations based on risk levels
- **Interactive Visualizations**: Beautiful charts and gauges for data exploration

## 🎯 Features

### 1. Single Prediction Page (`🔮 Single Prediction`)
- Interactive form for entering personal health data
- Real-time predictions for sleep quality (1-10 scale)
- Sleep disorder classification with confidence scores
- Risk level assessment (Low/Medium/High)
- Personalized health recommendations
- Visual gauges and charts for results

### 2. Batch Predictions Page (`📊 Batch Predictions`)
- Upload CSV files with multiple records
- Process bulk predictions efficiently
- Download results as CSV
- Summary statistics for batch analysis

### 3. Analytics Dashboard (`📈 Analytics`)
- Population-level health overview
- Sleep quality distribution
- Risk level distribution
- Age vs Sleep Quality scatter plot
- Sleep disorder distribution pie chart
- Key metrics and statistics

### 4. About Page (`ℹ️ About`)
- Detailed information about models and features
- Technical stack information
- Data privacy details
- Disclaimer and usage guidelines

## 📊 Models Used

### Sleep Quality Predictor
- **Type**: Regression Model (XGBoost/Random Forest)
- **Target**: Sleep Quality Score (1-10)
- **Features**: 28 engineered features
- **Performance Metric**: R² Score

### Sleep Disorder Classifier
- **Type**: Multi-class Classification (XGBoost/LightGBM)
- **Target Classes**: None, Insomnia, Sleep Apnea, Narcolepsy, REM Sleep Behavior Disorder
- **Features**: 28 engineered features
- **Performance Metric**: F1-Score
- **Confidence**: Probability-based confidence scores

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project
```bash
cd d:\GIT_HUB\12_Final_Projects_of_all\01_Analysis\main
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files
Ensure these files exist in the same directory as `streamlit_app.py`:
- `sleep_quality_model.pkl`
- `sleep_disorder_model.pkl`
- `disorder_label_encoder.pkl`

## 🚀 Running the Application

### Option 1: Run Locally
```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Option 2: Run with Custom Port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Option 3: Run in Headless Mode
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## 📝 Input Features

The application expects the following inputs:

### Personal Information
- **Age**: 18-80 years
- **Gender**: Male/Female
- **Occupation**: Manual Labor, Office Worker, Retired, Student

### Physical Health Metrics
- **BMI Category**: Underweight, Normal, Overweight, Obese
- **Systolic BP**: 80-180 mmHg
- **Diastolic BP**: 50-120 mmHg
- **Heart Rate**: 40-140 bpm

### Sleep Information
- **Sleep Duration**: 2-12 hours
- **Sleep Efficiency**: 0-100%

### Activity Level
- **Physical Activity Level**: 0-150 min/day
- **Daily Steps**: 1,000-50,000 steps
- **Activity Category**: Sedentary, Low_Active, Somewhat_Active, Active, Very_Active

### Stress & Categorical
- **Stress Level**: 1-10 scale
- **Various Category Fields**: Sleep Duration Category, BP Category, Heart Rate Category, Steps Category

## 📊 Output Predictions

### Sleep Quality Prediction
- **Output**: Score from 1-10
- **Interpretation**: 
  - 1-3: Poor
  - 4-5: Fair
  - 6-7: Good
  - 8-10: Excellent

### Sleep Disorder Prediction
- **Output**: Classification with confidence percentage
- **Possible Disorders**: None, Insomnia, Sleep Apnea, Narcolepsy, REM Sleep Behavior Disorder
- **Confidence Score**: 0-100% confidence level

### Risk Assessment
- **Low Risk**: Confidence ≤ 50%
- **Medium Risk**: Confidence 50-75%
- **High Risk**: Confidence > 75%

## 📁 File Structure

```
01_Analysis/main/
├── streamlit_app.py                 # Main Streamlit application
├── 05_final_model.py               # Model deployment script
├── 04_ml_model.ipynb               # ML model training notebook
├── sleep_quality_model.pkl         # Trained regression model
├── sleep_disorder_model.pkl        # Trained classification model
├── disorder_label_encoder.pkl      # Label encoder for disorders
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── Dataset/
    ├── sleep_health_ml_ready_full.csv
    ├── sleep_health_processed_for_viz.csv
    ├── sleep_health_with_predictions.csv
    ├── feature_names_quality.csv
    └── feature_names_disorder.csv
```

## 🔄 Batch Prediction Format

For batch predictions, upload a CSV file with the following columns:

```csv
Age,Sleep Duration,Physical Activity Level,Stress Level,Heart Rate,Daily Steps,SleepDisorder_Imputed,Systolic_BP,Diastolic_BP,Sleep_Efficiency,Health_Risk_Score,BMI Category_Encoded,Sleep_Duration_Category_Encoded,Activity_Category_Encoded,Stress_Category_Encoded,BP_Category_Encoded,Gender_Male,Occupation_Office Worker,Occupation_Retired,Occupation_Student,Age_Group_Middle_Age,Age_Group_Senior,Age_Group_Young_Adult,Heart_Rate_Category_Normal,Steps_Category_Low_Active,Steps_Category_Sedentary,Steps_Category_Somewhat_Active
35,7.0,50,5,70,10000,1,120,80,80,6.3,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1
```

## 🎨 Customization

### Change Theme
Edit the `st.set_page_config()` call in the app:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="😴",
    theme="dark"  # Change to "dark" or "light"
)
```

### Modify Colors
Edit the CSS in the custom_css variable to change colors and styling.

### Add New Features
Add new sections to the main() function with additional prediction capabilities.

## ⚙️ Model Information

### Training Data
- **Records**: 402 individuals
- **Features**: 28 engineered features
- **Target Variables**: 
  - Sleep Quality (1-10, continuous)
  - Sleep Disorder (categorical)

### Feature Engineering
- One-hot encoding for categorical variables
- Feature scaling for numerical variables
- Creation of derived features (Age Groups, Categories)
- Handling of missing values

### Class Distribution
- **None**: Majority class
- **Insomnia**: Common disorder
- **Sleep Apnea**: Common disorder
- **Others**: Rare classes

### Data Balancing
- SMOTE applied during training for disorder classification
- Stratified train-test split

## 📈 Performance Metrics

### Regression Model (Sleep Quality)
- **Metric**: R² Score, RMSE, MAE
- **Optimization**: Maximized R² Score

### Classification Model (Sleep Disorder)
- **Metric**: Accuracy, Precision, Recall, F1-Score
- **Optimization**: Maximized F1-Score
- **Class Balancing**: SMOTE applied

## 🔒 Security & Privacy

- ✅ No data is stored or logged
- ✅ All predictions are performed locally
- ✅ No external API calls
- ✅ No personal information transmitted
- ✅ Compliant with data privacy standards

## ⚠️ Disclaimer

**Important**: This application is for **educational and informational purposes only**.

- It should **NOT replace professional medical advice**
- Always consult with qualified healthcare providers for:
  - Sleep disorder diagnosis
  - Medical treatment recommendations
  - Medication suggestions
  - Lifestyle modifications

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'sleep_quality_model.pkl'"
**Solution**: Ensure all .pkl files are in the same directory as streamlit_app.py

### Issue: "ValueError: Prediction shape mismatch"
**Solution**: Ensure input data has exactly 28 features in the correct order

### Issue: Application runs slowly
**Solution**: 
- Close other applications
- Use batch processing instead of single predictions
- Increase available RAM

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the model documentation
3. Examine error messages for specific guidance
4. Contact the development team

## 📚 References

### Libraries Used
- **Streamlit**: https://streamlit.io/
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Plotly**: https://plotly.com/

### Machine Learning Concepts
- Regression Analysis
- Classification Algorithms
- Feature Engineering
- Cross-validation
- Class Imbalance Handling (SMOTE)

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Development

### Built With
- **Python 3.8+**
- **Streamlit 1.28+**
- **Scikit-learn 1.3+**
- **XGBoost 2.0+**
- **LightGBM 4.1+**
- **Plotly 5.17+**

### Version History
- **v1.0** (December 2024): Initial release
  - Single prediction functionality
  - Batch processing
  - Analytics dashboard
  - Model performance visualization

## 🎓 Educational Value

This project demonstrates:
- Machine Learning model deployment
- Web application development with Streamlit
- Data visualization techniques
- Model preprocessing and feature engineering
- User interface design best practices
- Real-world data science workflow

---

**Last Updated**: December 2024  
**Maintained By**: Data Science Team  
**Version**: 1.0.0
