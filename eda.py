"""
Deep EDA for Gridlock Hackathon 2.0
Analyzes temporal patterns, spatial clusters, and day-to-day drift.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train, test

def parse_time(df):
    ts = df["timestamp"].astype(str).str.split(":", expand=True)
    df["hour"] = ts[0].astype(int)
    df["minute"] = ts[1].astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    return df

def analyze_temporal_patterns(train):
    print("\n── Temporal Analysis ──")
    train = parse_time(train)
    
    # 1. Global Hourly Pattern
    hourly_mean = train.groupby(["day", "hour"])["demand"].mean().reset_index()
    print("Top 5 Hours by Demand (Day 48):")
    d48_hours = hourly_mean[hourly_mean["day"]==48].sort_values("demand", ascending=False).head(5)
    print(d48_hours[["hour", "demand"]])

    # 2. Day 49 vs Day 48 Comparison (Early Morning Only)
    d48_early = train[(train["day"]==48) & (train["time_slot"]<=8)]
    d49_early = train[(train["day"]==49) & (train["time_slot"]<=8)]
    
    if not d49_early.empty:
        ratio = d49_early["demand"].mean() / d48_early["demand"].mean()
        print(f"\nGlobal Early Morning Drift (D49/D48): {ratio:.3f}")
        
        # Per Geohash Drift Stability
        d48_geo = d48_early.groupby("geohash")["demand"].mean()
        d49_geo = d49_early.groupby("geohash")["demand"].mean()
        common = d48_geo.index.intersection(d49_geo.index)
        drifts = d49_geo[common] / d48_geo[common]
        print(f"Drift Stats: Mean={drifts.mean():.3f}, Std={drifts.std():.3f}")
        print(f"Drift Range: {drifts.min():.3f} to {drifts.max():.3f}")
        
        # Save drift distribution
        plt.figure(figsize=(10, 6))
        plt.hist(drifts, bins=50, color='steelblue', edgecolor='black')
        plt.title("Distribution of Geohash Drift (D49 Early / D48 Early)")
        plt.xlabel("Drift Ratio")
        plt.ylabel("Count")
        plt.savefig(os.path.join(OUTPUT_DIR, "drift_distribution.png"))
        print("Saved drift_distribution.png")

def analyze_spatial_clusters(train):
    print("\n── Spatial Analysis ──")
    geo_stats = train.groupby("geohash")["demand"].agg(["mean", "std", "count"]).reset_index()
    geo_stats.columns = ["geohash", "geo_mean", "geo_std", "geo_count"]
    
    # Identify High Variance vs Stable Locations
    high_var = geo_stats[geo_stats["geo_std"] > geo_stats["geo_std"].quantile(0.9)]
    stable = geo_stats[geo_stats["geo_std"] < geo_stats["geo_std"].quantile(0.1)]
    
    print(f"Total Geohashes: {len(geo_stats)}")
    print(f"High Variance (>90%ile): {len(high_var)}")
    print(f"Stable (<10%ile): {len(stable)}")
    
    # Check RoadType Distribution per Geohash (is it consistent?)
    road_consistency = train.groupby("geohash")["RoadType"].nunique()
    multi_road = road_consistency[road_consistency > 1]
    print(f"Geohashes with multiple RoadTypes: {len(multi_road)} (Noise in data?)")

def analyze_test_coverage(train, test):
    print("\n── Test Coverage Analysis ──")
    train = parse_time(train)
    test = parse_time(test)
    
    # Which time slots are in Test?
    test_slots = sorted(test["time_slot"].unique())
    train_slots = sorted(train["time_slot"].unique())
    
    print(f"Test Time Slots: {min(test_slots)} to {max(test_slots)}")
    print(f"Train Time Slots: {min(train_slots)} to {max(train_slots)}")
    
    # Overlap
    overlap = set(test_slots).intersection(set(train_slots))
    print(f"Overlapping Slots: {len(overlap)} / {len(test_slots)}")
    
    # Lag Coverage
    d48 = train[train["day"]==48][["geohash", "timestamp"]]
    d48_set = set(zip(d48["geohash"], d48["timestamp"]))
    
    test_keys = set(zip(test["geohash"], test["timestamp"]))
    matched = len(test_keys.intersection(d48_set))
    print(f"Lag Coverage (D48 match): {matched} / {len(test)} ({100*matched/len(test):.1f}%)")

def main():
    print("=" * 60)
    print("  DEEP EDA — Gridlock Hackathon 2.0")
    print("=" * 60)
    
    train, test = load_data()
    analyze_temporal_patterns(train)
    analyze_spatial_clusters(train)
    analyze_test_coverage(train, test)
    
    print("\nDeep EDA Complete.")

if __name__ == "__main__":
    main()