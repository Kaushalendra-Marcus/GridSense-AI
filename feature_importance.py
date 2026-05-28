"""
Gridlock Hackathon 2.0 — Feature Importance Viewer
====================================================
Run AFTER train.py:  python feature_importance.py
Prints top features from saved models.
"""

import os, pickle
import pandas as pd
import numpy as np

MODEL_DIR = "models"

def main():
    path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    if not os.path.exists(path):
        print("❌ No models found. Run train.py first.")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    models       = data["models"]
    feature_cols = data["feature_cols"]
    oof_score    = data["oof_score"]

    print(f"\nOOF Score: {oof_score:.4f} / 100\n")

    # Average importance across folds
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)

    df = (pd.DataFrame({"feature": feature_cols, "importance": importances})
          .sort_values("importance", ascending=False)
          .reset_index(drop=True))

    print("── Top 30 Features ──")
    print(df.head(30).to_string(index=False))

    # Save to CSV
    df.to_csv("outputs/feature_importance.csv", index=False)
    print("\nSaved → outputs/feature_importance.csv")

if __name__ == "__main__":
    main()