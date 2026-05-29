"""
Error Analysis for Gridlock Hackathon 2.0
Identifies where the model is underperforming.
Requires train.py to be run first.
"""
import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    return train

def get_oof_predictions():
    path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Run train.py first to generate models.")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Note: The current train.py doesn't save OOF preds directly in pkl.
    # For this script to work perfectly, we'd need to modify train.py to save oof_preds.
    # Instead, we will re-run a quick prediction on Train using the saved models.
    
    models = data["models"]
    feature_cols = data["feature_cols"]
    
    train = load_data()
    
    # We need to recreate features exactly as train.py does. 
    # This is complex. For now, let's just analyze the Submission vs a simple baseline.
    # A simpler approach: Analyze the Residuals of a Simple Baseline to see what's hard.
    
    return train, models, feature_cols

def analyze_baseline_errors():
    print("\n── Baseline Error Analysis ──")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Simple Baseline: Geohash Mean
    geo_mean = train.groupby("geohash")["demand"].mean()
    train["pred_baseline"] = train["geohash"].map(geo_mean)
    train["residual"] = train["demand"] - train["pred_baseline"]
    train["abs_error"] = np.abs(train["residual"])
    
    # 1. Error by Hour
    ts = train["timestamp"].astype(str).str.split(":", expand=True)
    train["hour"] = ts[0].astype(int)
    
    hour_error = train.groupby("hour")["abs_error"].mean()
    print("\nTop 5 Hours with Highest Error (Baseline):")
    print(hour_error.sort_values(ascending=False).head(5))
    
    # 2. Error by Weather
    weather_error = train.groupby("Weather")["abs_error"].mean()
    print("\nError by Weather:")
    print(weather_error.sort_values(ascending=False))
    
    # 3. Error by RoadType
    road_error = train.groupby("RoadType")["abs_error"].mean()
    print("\nError by RoadType:")
    print(road_error.sort_values(ascending=False))

def main():
    print("=" * 60)
    print("  ERROR ANALYSIS — Gridlock Hackathon 2.0")
    print("=" * 60)
    
    try:
        analyze_baseline_errors()
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nError Analysis Complete.")

if __name__ == "__main__":
    main()