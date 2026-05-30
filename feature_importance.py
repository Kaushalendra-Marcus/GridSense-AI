"""
Gridlock Hackathon 2.0 — Feature Importance Viewer
====================================================
Run AFTER train.py:  python feature_importance.py

Outputs
-------
- Top-N feature table (terminal)
- Feature importance by fold (variance check)
- outputs/feature_importance.csv
- outputs/feature_importance_by_fold.csv
"""

import os
import pickle
import numpy as np
import pandas as pd

MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
DIVIDER    = "=" * 70


def ascii_bar(val, max_val, width=40):
    filled = int(round(width * val / max_val)) if max_val > 0 else 0
    return "[" + "#" * filled + " " * (width - filled) + "]"


def main():
    path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    if not os.path.exists(path):
        print("[ERROR] No models found. Run train.py first.")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    models       = data["models"]
    feature_cols = data["feature_cols"]
    oof_score    = data["oof_score"]
    n_folds      = len(models)

    print(DIVIDER)
    print(f"  Feature Importance — {n_folds} folds")
    print(f"  OOF Score: {oof_score:.4f} / 100")
    print(DIVIDER)

    # ── Per-fold importance matrix ────────────────────────────────────
    imp_matrix = np.zeros((n_folds, len(feature_cols)))
    for i, model in enumerate(models):
        imp_matrix[i] = model.feature_importances_

    imp_mean = imp_matrix.mean(axis=0)
    imp_std  = imp_matrix.std(axis=0)

    df = pd.DataFrame({
        "feature"    : feature_cols,
        "importance" : imp_mean,
        "std"        : imp_std,
        "cv_pct"     : np.where(imp_mean > 0, 100 * imp_std / imp_mean, 0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["rank"]    = range(1, len(df) + 1)
    max_imp       = df["importance"].max()

    # ── Top-40 table ─────────────────────────────────────────────────
    top_n = min(40, len(df))
    print(f"\n  Top {top_n} Features (mean importance across folds):\n")
    print(f"  {'Rank':<5} {'Feature':<35} {'Importance':>10} {'Std':>8} "
          f"{'CV%':>6}  Bar")
    print("  " + "-" * 90)
    for _, row in df.head(top_n).iterrows():
        bar = ascii_bar(row["importance"], max_imp)
        print(f"  {int(row['rank']):<5} {row['feature']:<35} "
              f"{row['importance']:>10.1f} {row['std']:>8.1f} "
              f"{row['cv_pct']:>5.1f}%  {bar}")

    # ── Bottom 10 (near-zero importance — candidates to drop) ─────────
    print(f"\n  Bottom 10 Features (lowest importance — consider dropping):\n")
    print(f"  {'Rank':<5} {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 55)
    for _, row in df.tail(10).iterrows():
        print(f"  {int(row['rank']):<5} {row['feature']:<35} {row['importance']:>10.1f}")

    # ── High variance features (unstable across folds) ────────────────
    unstable = df[df["cv_pct"] > 30].sort_values("cv_pct", ascending=False)
    if len(unstable) > 0:
        print(f"\n  High-variance features (CV% > 30% — potentially noisy):\n")
        print(f"  {'Feature':<35} {'Importance':>10} {'CV%':>6}")
        print("  " + "-" * 55)
        for _, row in unstable.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:>10.1f} {row['cv_pct']:>5.1f}%")

    # ── Summary stats ─────────────────────────────────────────────────
    zero_imp = (df["importance"] == 0).sum()
    print(f"\n  Summary:")
    print(f"    Total features          : {len(df)}")
    print(f"    Zero-importance features: {zero_imp}  "
          f"{'(safe to drop)' if zero_imp > 0 else ''}")
    print(f"    Top-5 features account for "
          f"{100*df.head(5)['importance'].sum()/df['importance'].sum():.1f}% of total importance")
    print(f"    Top-10 features account for "
          f"{100*df.head(10)['importance'].sum()/df['importance'].sum():.1f}% of total importance")

    # ── Per-fold importance table (for top 20) ────────────────────────
    top20 = df.head(20)["feature"].tolist()
    fold_rows = []
    for i in range(n_folds):
        row = {"fold": i + 1}
        for j, feat in enumerate(feature_cols):
            if feat in top20:
                row[feat] = imp_matrix[i, j]
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    print(f"\n  Per-fold importance for top-20 features:")
    print(fold_df[top20].round(1).to_string())

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    fold_path = os.path.join(OUTPUT_DIR, "feature_importance_by_fold.csv")
    df.to_csv(main_path, index=False)
    fold_df.reset_index().to_csv(fold_path, index=False)
    print(f"\n  Saved -> {main_path}")
    print(f"  Saved -> {fold_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
