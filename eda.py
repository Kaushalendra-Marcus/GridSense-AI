"""
Gridlock Hackathon 2.0 — Exploratory Data Analysis
====================================================
Run:  python eda.py
Prints key stats and checks the Day-48 lag coverage.
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "data"

def main():
    print("=" * 55)
    print("  EDA — Traffic Demand Prediction")
    print("=" * 55)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"\nTrain shape : {train.shape}")
    print(f"Test  shape : {test.shape}")

    print("\n── Train columns ──")
    print(train.dtypes)

    print("\n── Train head ──")
    print(train.head(3).to_string())

    print("\n── Missing values (train) ──")
    print(train.isnull().sum())

    print("\n── Target (demand) stats ──")
    print(train["demand"].describe())

    print("\n── Day range ──")
    print(f"  Train days : {sorted(train['day'].unique())}")
    if "day" in test.columns:
        print(f"  Test  days : {sorted(test['day'].unique())}")

    # ── Golden feature check ──────────────────────────
    print("\n── Day-48 lag coverage check ──")
    max_train_day = train["day"].max()
    print(f"  Max training day: {max_train_day}")

    if "day" in test.columns:
        test_days = sorted(test["day"].unique())
        print(f"  Test days: {test_days}")

        # Check overlap of geohash × timestamp between last train day & test
        last_day_data  = train[train["day"] == max_train_day]
        overlap_keys   = pd.merge(
            test[["geohash","timestamp"]],
            last_day_data[["geohash","timestamp"]],
            on=["geohash","timestamp"], how="inner"
        )
        pct = len(overlap_keys) / len(test) * 100
        print(f"  Test rows that match Day-{max_train_day} (geohash+timestamp): "
              f"{len(overlap_keys)} / {len(test)} ({pct:.1f}%)")
        if pct > 80:
            print("  ✅ HIGH overlap → demand_d48 will be VERY powerful")
        elif pct > 30:
            print("  ⚠️  MEDIUM overlap → lag helps but needs fallback")
        else:
            print("  ❌ LOW overlap → lag feature has limited value; use target encoding")
    else:
        print("  No 'day' column in test — cannot check lag coverage")

    # ── Timestamp format ─────────────────────────────
    print("\n── Timestamp samples ──")
    print(train["timestamp"].value_counts().head(5))

    # ── Geohash stats ─────────────────────────────────
    print(f"\n── Unique geohashes ──")
    print(f"  Train: {train['geohash'].nunique()}")
    print(f"  Test : {test['geohash'].nunique()}")
    unseen = set(test["geohash"].unique()) - set(train["geohash"].unique())
    print(f"  Unseen in test: {len(unseen)}")

    # ── Categorical value counts ──────────────────────
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        if col in train.columns:
            print(f"\n── {col} ──")
            print(train[col].value_counts().head(5))

    print("\nEDA done.")

if __name__ == "__main__":
    main()