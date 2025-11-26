# ⚡ COMMAND REFERENCE GUIDE

## 🚀 Quick Commands

### Installation & Setup
```bash
# Navigate to project directory
cd d:\GIT_HUB\12_Final_Projects_of_all\01_Analysis\main

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
pip list | grep pandas
```

### Running the Application
```bash
# Basic launch
streamlit run streamlit_app.py

# Custom port
streamlit run streamlit_app.py --server.port 8502

# Debug mode
streamlit run streamlit_app.py --logger.level=debug

# Headless mode (no browser)
streamlit run streamlit_app.py --server.headless true

# Remote access
streamlit run streamlit_app.py --server.enableXsrfProtection=false
```

### Development Commands
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Update all packages
pip install --upgrade -r requirements.txt

# Create requirements from current environment
pip freeze > requirements.txt

# Virtual environment cleanup
deactivate  # Exit venv
rmdir venv  # Remove directory (Windows)
rm -rf venv  # Remove directory (Linux/Mac)
```

### Troubleshooting
```bash
# Clear Streamlit cache
streamlit cache clear

# View logs
streamlit logs

# Check port usage
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # macOS/Linux

# Kill process on port
taskkill /PID <PID> /F  # Windows
kill -9 <PID>  # macOS/Linux
```

---

## 📂 File Management

### Directory Structure
```bash
# List all files
dir  # Windows
ls -la  # Linux/Mac

# Copy model files
copy "source_path\*.pkl" .

# Check file sizes
dir /s  # Windows
du -sh  # Linux/Mac
```

### CSV Operations
```bash
# View CSV structure
python -c "import pandas as pd; print(pd.read_csv('file.csv').head())"

# Check CSV columns
python -c "import pandas as pd; print(pd.read_csv('file.csv').columns.tolist())"

# Verify records count
python -c "import pandas as pd; print(len(pd.read_csv('file.csv')))"
```

---

## 🔧 Model Operations

### Load and Test Models
```python
import pickle
import pandas as pd

# Load models
with open('sleep_quality_model.pkl', 'rb') as f:
    quality_model = pickle.load(f)

with open('sleep_disorder_model.pkl', 'rb') as f:
    disorder_model = pickle.load(f)

with open('disorder_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

print("✓ All models loaded successfully!")
print(f"Disorder classes: {list(encoder.classes_)}")
```

### Make Predictions
```python
import numpy as np

# Create sample input (28 features)
X_sample = np.random.rand(1, 28)

# Predict quality
quality_pred = quality_model.predict(X_sample)[0]
print(f"Quality: {quality_pred:.2f}")

# Predict disorder
disorder_pred = disorder_model.predict(X_sample)[0]
disorder_class = encoder.inverse_transform([disorder_pred])[0]
print(f"Disorder: {disorder_class}")

# Get confidence
confidence = disorder_model.predict_proba(X_sample)[0].max() * 100
print(f"Confidence: {confidence:.2f}%")
```

---

## 🧪 Testing Commands

### Test Application Locally
```bash
# Run and keep browser open
streamlit run streamlit_app.py --client.showErrorDetails=true

# Test with sample data
cd "Dataset"
python -c "import pandas as pd; df = pd.read_csv('sleep_health_processed_for_viz.csv'); print(df.info())"
```

### Performance Testing
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Profile application
python -m cProfile -s cumulative streamlit_app.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler streamlit_app.py
```

---

## 📊 Data Analysis Commands

### Exploratory Data Analysis
```python
import pandas as pd
import numpy as np

# Load and explore data
df = pd.read_csv('sleep_health_with_predictions.csv')

# Basic statistics
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Check predictions
print(f"Quality range: {df['Predicted_Sleep_Quality'].min():.2f} - {df['Predicted_Sleep_Quality'].max():.2f}")
print(f"Disorder distribution:\n{df['Predicted_Disorder'].value_counts()}")
print(f"Risk distribution:\n{df['Risk_Level'].value_counts()}")
```

### Data Validation
```python
# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values found:")
    print(missing[missing > 0])

# Check data types
print(df.dtypes)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Check feature ranges
print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"Sleep Duration range: {df['Sleep Duration'].min()} - {df['Sleep Duration'].max()}")
```

---

## 🔄 Batch Processing Commands

### Process Large CSV Files
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Read in chunks for large files
chunk_size = 100
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = process_chunk(chunk)
    chunks.append(processed)

result = pd.concat(chunks, ignore_index=True)
result.to_csv('processed_output.csv', index=False)
```

### Generate Predictions for Dataset
```python
import pandas as pd
import pickle

# Load models and data
with open('sleep_quality_model.pkl', 'rb') as f:
    quality_model = pickle.load(f)

with open('sleep_disorder_model.pkl', 'rb') as f:
    disorder_model = pickle.load(f)

with open('disorder_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load data
df = pd.read_csv('Dataset/sleep_health_processed_for_viz.csv')

# Get feature columns (28)
feature_cols = [...]  # List of 28 feature names

X = df[feature_cols]

# Make predictions
df['Predicted_Quality'] = quality_model.predict(X)
df['Predicted_Disorder'] = encoder.inverse_transform(disorder_model.predict(X))
df['Confidence'] = disorder_model.predict_proba(X).max(axis=1) * 100

# Save results
df.to_csv('predictions.csv', index=False)
print("✓ Predictions saved!")
```

---

## 📝 Logging & Debugging

### Enable Detailed Logging
```bash
# Set environment variables
set STREAMLIT_LOGGER_LEVEL=debug  # Windows
export STREAMLIT_LOGGER_LEVEL=debug  # Linux/Mac

# Run with debug
streamlit run streamlit_app.py --logger.level=debug
```

### Create Log File
```python
import logging

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Application started")
```

---

## 🌐 Deployment Commands

### Docker (If using containerization)
```bash
# Build image
docker build -t sleep-health:latest .

# Run container
docker run -p 8501:8501 sleep-health:latest

# Deploy to registry
docker push your-registry/sleep-health:latest
```

### Streamlit Cloud
```bash
# Deploy to Streamlit Cloud
# 1. Push to GitHub
git push origin main

# 2. Go to https://share.streamlit.io
# 3. Connect GitHub account
# 4. Select repository and branch
# 5. Deploy!
```

### Heroku Deployment
```bash
# Install Heroku CLI
npm install -g heroku

# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

---

## 🔐 Security & Cleanup

### Remove Sensitive Data
```bash
# Remove all __pycache__ directories
python -c "import shutil; import os; [shutil.rmtree(f'{root}/__pycache__') for root, dirs, _ in os.walk('.') if '__pycache__' in dirs]"

# Remove .pyc files
find . -name "*.pyc" -delete  # Linux/Mac
for /r %d in (__pycache__) do @if exist "%d" rd /s /q "%d"  # Windows
```

### Backup Important Files
```bash
# Create backup
copy streamlit_app.py streamlit_app.backup.py

# Archive project
tar -czf sleep-health-backup.tar.gz .  # Linux/Mac
```

---

## 📊 Useful Python Commands

### Interactive Testing
```bash
# Start Python interpreter
python

# Or use iPython
ipython

# Or use Jupyter
jupyter notebook
```

### Inside Python/iPython
```python
# Import and test
import streamlit as st
import pandas as pd
import numpy as np

# Load models
import pickle
models = pickle.load(open('sleep_quality_model.pkl', 'rb'))

# Test prediction
test_input = np.random.rand(1, 28)
prediction = models.predict(test_input)
print(prediction)

# Exit
exit()
```

---

## 🆘 Emergency Commands

### If Application Crashes
```bash
# Kill all Python processes
taskkill /F /IM python.exe  # Windows

# Clear Streamlit cache
streamlit cache clear

# Clear browser cache
# Manual: Settings > Privacy > Clear browsing data

# Restart application
streamlit run streamlit_app.py --client.caching.maxMessageSize 200
```

### Reset Environment
```bash
# Deactivate and delete venv
deactivate
rmdir /s /q venv  # Windows
rm -rf venv  # Linux/Mac

# Reinstall from scratch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📚 Documentation Commands

### Generate Documentation
```bash
# Generate API docs (if using pdoc)
pip install pdoc
pdoc streamlit_app.py

# Generate requirements documentation
pip-compile requirements.txt

# View help
python -m streamlit --help
```

---

## ⚡ Performance Optimization

### Profile Application
```python
import cProfile
import pstats
import streamlit as st

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

### Monitor Memory
```python
import psutil

process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
print(f"CPU usage: {process.cpu_percent()}%")
```

---

## 🎯 Common Workflows

### Complete Workflow: From Setup to Running
```bash
# 1. Clone/Navigate
cd d:\GIT_HUB\12_Final_Projects_of_all\01_Analysis\main

# 2. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Verify
pip list

# 4. Run
streamlit run streamlit_app.py

# 5. Access
# Opens at http://localhost:8501
```

### Workflow: Update and Redeploy
```bash
# 1. Make changes
# Edit streamlit_app.py

# 2. Test locally
streamlit run streamlit_app.py

# 3. Update requirements if needed
pip freeze > requirements.txt

# 4. Commit changes
git add .
git commit -m "Update: feature description"
git push origin main

# 5. Auto-deploy on Streamlit Cloud
# (if connected to GitHub)
```

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Complete ✅
