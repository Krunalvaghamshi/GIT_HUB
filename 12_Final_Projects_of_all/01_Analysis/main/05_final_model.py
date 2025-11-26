"""
SLEEP HEALTH & DISORDER PREDICTION - FINAL MODEL DEPLOYMENT
============================================================
This script loads the trained ML models and provides a complete interface for:
1. Predicting Sleep Quality (Regression)
2. Predicting Sleep Disorders (Classification)
3. Risk Assessment
4. Batch Predictions
5. Real-time Predictions

Models Used:
- sleep_quality_model.pkl: Random Forest/XGBoost for Quality Prediction
- sleep_disorder_model.pkl: XGBoost/LightGBM for Disorder Classification
- disorder_label_encoder.pkl: Label Encoder for disorder classes

Author: Data Science Team
Date: 2024
"""

import pickle
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# 1. MODEL LOADER CLASS
# ============================================================

class SleepHealthModelLoader:
    """Load and manage trained sleep health models"""
    
    def __init__(self, model_dir='.'):
        """
        Initialize model loader
        
        Args:
            model_dir (str): Directory containing model pickle files
        """
        self.model_dir = model_dir
        self.quality_model = None
        self.disorder_model = None
        self.disorder_encoder = None
        self.is_loaded = False
        
    def load_models(self):
        """Load all models from pickle files"""
        try:
            # Load Quality Prediction Model
            quality_model_path = os.path.join(self.model_dir, 'sleep_quality_model.pkl')
            with open(quality_model_path, 'rb') as f:
                self.quality_model = pickle.load(f)
            print("✓ Sleep Quality model loaded successfully")
            
            # Load Disorder Prediction Model
            disorder_model_path = os.path.join(self.model_dir, 'sleep_disorder_model.pkl')
            with open(disorder_model_path, 'rb') as f:
                self.disorder_model = pickle.load(f)
            print("✓ Sleep Disorder model loaded successfully")
            
            # Load Label Encoder
            encoder_path = os.path.join(self.model_dir, 'disorder_label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.disorder_encoder = pickle.load(f)
            print("✓ Disorder Label Encoder loaded successfully")
            
            self.is_loaded = True
            print("\n✓ All models loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model file not found - {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_disorder_classes(self):
        """Get available disorder classes"""
        if self.disorder_encoder is not None:
            return list(self.disorder_encoder.classes_)
        return None


# ============================================================
# 2. PREDICTION ENGINE
# ============================================================

class SleepHealthPredictor:
    """Make predictions using trained models"""
    
    def __init__(self, model_loader):
        """
        Initialize predictor
        
        Args:
            model_loader (SleepHealthModelLoader): Loaded model loader instance
        """
        self.loader = model_loader
        if not model_loader.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
    
    def predict_sleep_quality(self, features_df):
        """
        Predict sleep quality score (regression)
        
        Args:
            features_df (pd.DataFrame): Features dataframe with all required columns
            
        Returns:
            np.array: Predicted sleep quality scores
        """
        if self.loader.quality_model is None:
            raise ValueError("Quality model not loaded")
        
        predictions = self.loader.quality_model.predict(features_df)
        return predictions
    
    def predict_sleep_disorder(self, features_df, return_probabilities=False):
        """
        Predict sleep disorder (classification)
        
        Args:
            features_df (pd.DataFrame): Features dataframe with all required columns
            return_probabilities (bool): If True, return prediction probabilities
            
        Returns:
            tuple: (predicted_labels, predicted_classes) or (predicted_labels, predicted_classes, probabilities)
        """
        if self.loader.disorder_model is None:
            raise ValueError("Disorder model not loaded")
        
        # Get predictions
        predicted_labels = self.loader.disorder_model.predict(features_df)
        predicted_classes = self.loader.disorder_encoder.inverse_transform(predicted_labels)
        
        if return_probabilities:
            probabilities = self.loader.disorder_model.predict_proba(features_df)
            return predicted_labels, predicted_classes, probabilities
        
        return predicted_labels, predicted_classes
    
    def get_prediction_confidence(self, features_df):
        """
        Get confidence scores for disorder predictions
        
        Args:
            features_df (pd.DataFrame): Features dataframe
            
        Returns:
            np.array: Confidence scores (0-100)
        """
        if self.loader.disorder_model is None:
            raise ValueError("Disorder model not loaded")
        
        probabilities = self.loader.disorder_model.predict_proba(features_df)
        confidence = probabilities.max(axis=1) * 100
        return confidence
    
    def get_risk_level(self, confidence_scores):
        """
        Categorize risk level based on confidence scores
        
        Args:
            confidence_scores (np.array or pd.Series): Confidence scores (0-100)
            
        Returns:
            np.array: Risk levels ('Low_Risk', 'Medium_Risk', 'High_Risk')
        """
        risk_levels = pd.cut(
            confidence_scores,
            bins=[0, 50, 75, 100],
            labels=['Low_Risk', 'Medium_Risk', 'High_Risk'],
            right=True
        )
        return risk_levels.values


# ============================================================
# 3. BATCH PREDICTION
# ============================================================

class BatchPredictor:
    """Process batch predictions from CSV files"""
    
    def __init__(self, predictor):
        """
        Initialize batch predictor
        
        Args:
            predictor (SleepHealthPredictor): Predictor instance
        """
        self.predictor = predictor
    
    def predict_from_csv(self, csv_path, feature_columns=None, output_path=None):
        """
        Load data from CSV and generate predictions
        
        Args:
            csv_path (str): Path to input CSV file
            feature_columns (list): List of feature column names (if None, auto-detect)
            output_path (str): Path to save predictions (if None, auto-generate)
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded data from {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            if feature_columns is None:
                # Auto-detect feature columns (all except known target columns)
                exclude_cols = ['Quality of Sleep', 'Sleep Disorder', 'Person ID', 'Gender', 'Occupation']
                feature_columns = [col for col in df.columns if col not in exclude_cols]
                print(f"✓ Auto-detected {len(feature_columns)} feature columns")
            
            # Extract features
            X = df[feature_columns].copy()
            
            # Generate predictions
            print("Generating predictions...")
            
            # Quality prediction
            quality_pred = self.predictor.predict_sleep_quality(X)
            df['Predicted_Sleep_Quality'] = quality_pred
            
            # Disorder prediction
            disorder_labels, disorder_classes = self.predictor.predict_sleep_disorder(X)
            df['Predicted_Disorder'] = disorder_classes
            
            # Confidence and risk
            confidence = self.predictor.get_prediction_confidence(X)
            df['Disorder_Confidence'] = confidence
            df['Risk_Level'] = self.predictor.get_risk_level(confidence)
            
            print(f"✓ Predictions generated successfully")
            
            # Save results
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"predictions_{timestamp}.csv"
            
            df.to_csv(output_path, index=False)
            print(f"✓ Results saved to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing batch predictions: {e}")
            return None


# ============================================================
# 4. SINGLE PREDICTION
# ============================================================

class InteractivePredictor:
    """Interactive single prediction interface"""
    
    def __init__(self, predictor, feature_columns=None):
        """
        Initialize interactive predictor
        
        Args:
            predictor (SleepHealthPredictor): Predictor instance
            feature_columns (list): List of feature column names
        """
        self.predictor = predictor
        self.feature_columns = feature_columns or []
    
    def predict_from_dict(self, data_dict):
        """
        Predict from a dictionary of features
        
        Args:
            data_dict (dict): Dictionary with feature names as keys
            
        Returns:
            dict: Predictions and confidence scores
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data_dict])
            
            # Extract features in correct order
            X = df[self.feature_columns]
            
            # Quality prediction
            quality = self.predictor.predict_sleep_quality(X)[0]
            
            # Disorder prediction
            labels, classes = self.predictor.predict_sleep_disorder(X)
            disorder = classes[0]
            
            # Confidence
            confidence = self.predictor.get_prediction_confidence(X)[0]
            risk = self.predictor.get_risk_level([confidence])[0]
            
            results = {
                'Predicted_Sleep_Quality': round(quality, 2),
                'Predicted_Disorder': disorder,
                'Confidence': round(confidence, 2),
                'Risk_Level': risk
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None


# ============================================================
# 5. DEMO & USAGE
# ============================================================

def main():
    """Main execution example"""
    
    print("=" * 80)
    print("SLEEP HEALTH PREDICTION SYSTEM - FINAL MODEL")
    print("=" * 80)
    
    # Initialize model loader
    print("\n[1/5] Loading Models...")
    loader = SleepHealthModelLoader(model_dir='.')
    if not loader.load_models():
        print("Cannot proceed without models.")
        return
    
    # Initialize predictor
    print("\n[2/5] Initializing Predictor...")
    try:
        predictor = SleepHealthPredictor(loader)
        print("✓ Predictor initialized")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Display available classes
    print("\n[3/5] System Configuration:")
    disorder_classes = loader.get_disorder_classes()
    print(f"Available Sleep Disorders: {', '.join(disorder_classes)}")
    
    # Example batch prediction
    print("\n[4/5] Checking for input data...")
    csv_file = 'D:\\GIT_HUB\\12_Final_Projects_of_all\\01_Analysis\\Dataset\\sleep_health_with_predictions.csv'
    
    if os.path.exists(csv_file):
        print(f"Found dataset: {csv_file}")
        
        batch_pred = BatchPredictor(predictor)
        df_results = batch_pred.predict_from_csv(csv_file)
        
        if df_results is not None:
            print("\nSample Predictions:")
            print(df_results[['Person ID', 'Predicted_Sleep_Quality', 
                             'Predicted_Disorder', 'Risk_Level']].head(10))
    
    print("\n[5/5] System Ready!")
    print("\nUsage Examples:")
    print("  1. Batch Predictions: Use BatchPredictor.predict_from_csv()")
    print("  2. Single Prediction: Use InteractivePredictor.predict_from_dict()")
    print("  3. Direct Access: Use predictor.predict_sleep_quality() / predict_sleep_disorder()")
    
    print("\n" + "=" * 80)
    print("MODEL DEPLOYMENT READY")
    print("=" * 80)


# ============================================================
# 6. HELPER UTILITIES
# ============================================================

def validate_features(df, required_features):
    """Validate that all required features are present"""
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        return False
    print(f"✓ All {len(required_features)} features present")
    return True


def print_prediction_report(person_id, actual_quality, predicted_quality, 
                           actual_disorder, predicted_disorder, confidence, risk):
    """Print formatted prediction report"""
    print("\n" + "=" * 60)
    print(f"PREDICTION REPORT - Person ID: {person_id}")
    print("=" * 60)
    print(f"Sleep Quality:")
    print(f"  Actual:    {actual_quality}")
    print(f"  Predicted: {predicted_quality:.2f}")
    print(f"\nSleep Disorder:")
    print(f"  Actual:    {actual_disorder}")
    print(f"  Predicted: {predicted_disorder}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Risk Level: {risk}")
    print("=" * 60)


if __name__ == "__main__":
    main()
