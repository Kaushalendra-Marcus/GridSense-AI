"""
Feature Correlation Analysis
Checks for redundant features.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

def load_and_engineer():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Basic Temporal
    ts = train["timestamp"].astype(str).str.split(":", expand=True)
    train["hour"] = ts[0].astype(int)
    train["minute"] = ts[1].astype(int)
    train["time_slot"] = train["hour"] * 4 + train["minute"] // 15
    
    # Basic Geo
    train["geo_p4"] = train["geohash"].astype(str).str[:4]
    
    return train

def main():
    print("=" * 60)
    print("  FEATURE CORRELATION — Gridlock Hackathon 2.0")
    print("=" * 60)
    
    train = load_and_engineer()
    
    # Select numeric columns for correlation
    num_cols = ["demand", "hour", "time_slot", "NumberofLanes", "Temperature"]
    
    # Add Target Encodings roughly (for correlation check)
    geo_mean = train.groupby("geohash")["demand"].mean()
    train["geo_mean_enc"] = train["geohash"].map(geo_mean)
    
    slot_mean = train.groupby("time_slot")["demand"].mean()
    train["slot_mean_enc"] = train["time_slot"].map(slot_mean)
    
    num_cols += ["geo_mean_enc", "slot_mean_enc"]
    
    corr_matrix = train[num_cols].corr()
    
    print("\nCorrelation with Demand:")
    print(corr_matrix["demand"].sort_values(ascending=False))
    
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    print("\nSaved correlation_heatmap.png")

if __name__ == "__main__":
    main()