"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction
====================================================
Run:  python train.py
Output: outputs/submission.csv  +  models/*.pkl
"""

import os, warnings, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"
N_FOLDS     = 5
SEED        = 42

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      5000,
    "learning_rate":     0.02,
    "num_leaves":        255,
    "max_depth":         -1,
    "min_child_samples": 15,
    "subsample":         0.75,
    "subsample_freq":    1,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "min_split_gain":    0.001,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("\n[1/6] Loading data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test      = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Raw train: {train_raw.shape}  |  Test: {test.shape}")

    # Test is Day 49 only. Train on Day 49 rows; Day 48 is used only for lag.
    train = train_raw[train_raw["day"] == 49].reset_index(drop=True)
    print(f"  Training on Day 49 only: {train.shape}")
    return train_raw, train, test


# ─────────────────────────────────────────────
# STEP 2 — TEMPORAL FEATURES
# ─────────────────────────────────────────────
def add_temporal_features(df):
    """Extract and encode time-based features."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"]      = ts.dt.hour
    df["minute"]    = ts.dt.minute
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15   # 0-95 (15-min buckets)
    df["is_peak"]   = df["hour"].isin([7,8,9,17,18,19]).astype(int)
    df["is_night"]  = df["hour"].between(22, 23) | df["hour"].between(0, 5)
    df["is_night"]  = df["is_night"].astype(int)

    # Cyclical (midnight ↔ 23:45 continuity)
    df["sin_hour"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_timeslot"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["cos_timeslot"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["sin_day"]      = np.sin(2 * np.pi * df["day"] / 7)
    df["cos_day"]      = np.cos(2 * np.pi * df["day"] / 7)

    return df


# ─────────────────────────────────────────────
# STEP 3 — GEOHASH FEATURES
# ─────────────────────────────────────────────
def add_geo_features(df):
    """Decode geohash and extract spatial hierarchy."""
    gh = df["geohash"].astype(str)
    df["geo_p3"] = gh.str[:3]
    df["geo_p4"] = gh.str[:4]
    df["geo_p5"] = gh.str[:5]
    df["geo_p6"] = gh.str[:6]

    # Try to decode lat/lon (needs pygeohash — optional, won't crash if missing)
    try:
        import pygeohash as pgh
        coords = gh.apply(lambda x: pgh.decode(x) if len(x) >= 4 else (np.nan, np.nan))
        df["lat"] = coords.apply(lambda x: x[0])
        df["lon"] = coords.apply(lambda x: x[1])
    except ImportError:
        print("  [INFO] pygeohash not found — skipping lat/lon decode")
        df["lat"] = np.nan
        df["lon"] = np.nan

    return df


# ─────────────────────────────────────────────
# STEP 2b — FILL MISSING VALUES
# ─────────────────────────────────────────────
def fill_missing(train, test, train_raw=None):
    """Handle nulls: RoadType (600), Temperature (2495), Weather (797).
    If train_raw is provided, use it for computing fill statistics
    so Day 49-only train doesn't lose context from Day 48.
    """
    src = train_raw if train_raw is not None else train

    # Categorical: fill with mode per geohash prefix, then global mode
    for col in ["RoadType", "Weather"]:
        for df in [train, test]:
            geo_mode = (
                src.dropna(subset=[col])
                .groupby("geo_p4")[col]
                .agg(lambda x: x.mode()[0] if len(x) > 0 else np.nan)
            )
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "geo_p4"].map(geo_mode)
            global_mode = src[col].mode()[0]
            df[col].fillna(global_mode, inplace=True)

    # Numeric: fill Temperature with geohash+day median, then global median
    for df in [train, test]:
        geo_day_med = (
            src.dropna(subset=["Temperature"])
            .groupby(["geohash", "day"])["Temperature"].median()
        )
        df["Temperature"] = df["Temperature"].fillna(
            df.apply(lambda r: geo_day_med.get((r["geohash"], r["day"]), np.nan), axis=1)
        )
        global_temp_med = src["Temperature"].median()
        df["Temperature"].fillna(global_temp_med, inplace=True)

    return train, test


# ─────────────────────────────────────────────
# STEP 4 — LAG FEATURE
# ─────────────────────────────────────────────
def add_lag_features(train_raw, train, test):
    """
    Day 48 rows from train_raw are used purely as a lag source.
    train (Day 49 only) and test (Day 49) both get demand_d48
    from the matching geohash + timestamp in Day 48.
    """
    print("\n[3/6] Building lag features...")

    day48 = (train_raw[train_raw["day"] == 48][["geohash", "timestamp", "demand"]]
             .rename(columns={"demand": "demand_d48"}))

    # Merge lag onto Day 49 train and test
    train = train.merge(day48[["geohash", "timestamp", "demand_d48"]],
                        on=["geohash", "timestamp"], how="left")
    test  = test.merge(day48[["geohash", "timestamp", "demand_d48"]],
                       on=["geohash", "timestamp"], how="left")

    train_cov = train["demand_d48"].notna().mean() * 100
    test_cov  = test["demand_d48"].notna().mean()  * 100
    print(f"  Lag coverage  train: {train_cov:.1f}%  |  test: {test_cov:.1f}%")

    # Fill NaN lag with per-geohash median demand from Day 48
    geo_median = day48.groupby("geohash")["demand_d48"].median()
    train["demand_d48"] = train["demand_d48"].fillna(train["geohash"].map(geo_median))
    test["demand_d48"]  = test["demand_d48"].fillna(test["geohash"].map(geo_median))

    # Fallback 2: per-geo_p4-prefix + timestamp median
    day48["geo_p4"] = day48["geohash"].astype(str).str[:4]
    geo_p4_ts_med = day48.groupby(["geo_p4", "timestamp"])["demand_d48"].median()
    for df in [train, test]:
        mask = df["demand_d48"].isna()
        if mask.any():
            df.loc[mask, "demand_d48"] = df.loc[mask].apply(
                lambda r: geo_p4_ts_med.get((r["geohash"][:4], r["timestamp"]), np.nan), axis=1
            )

    # Fallback 3: per-timestamp median across all Day 48
    ts_med = day48.groupby("timestamp")["demand_d48"].median()
    for df in [train, test]:
        mask = df["demand_d48"].isna()
        if mask.any():
            df.loc[mask, "demand_d48"] = df.loc[mask, "timestamp"].map(ts_med)

    # Final fallback: global median
    global_med = day48["demand_d48"].median()
    train["demand_d48"].fillna(global_med, inplace=True)
    test["demand_d48"].fillna(global_med, inplace=True)

    print(f"  After fallback — NaN in train lag: {train['demand_d48'].isna().sum()}")
    print(f"  After fallback — NaN in test  lag: {test['demand_d48'].isna().sum()}")

    return train, test


# ─────────────────────────────────────────────
# STEP 5 — TARGET ENCODING (fold-safe)
# ─────────────────────────────────────────────
def add_target_encoding(train, test, kf):
    """
    Encode geohash × time_slot interaction using OOF strategy.
    Also encode geohash, geo_p4, day × time_slot.
    """
    print("\n[4/6] Target encoding (OOF)...")
    target = train["demand"]

    encode_keys = [
        "geohash",
        "geo_p4",
        ("geohash",   "time_slot"),   # location x time: key feature
        ("geohash",   "day"),
        ("geo_p4",    "time_slot"),
        ("day",       "time_slot"),
        ("RoadType",  "time_slot"),   # road type behaves differently at peak vs off-peak
        ("geohash",   "is_peak"),     # location-specific peak demand signature
        ("Weather",   "time_slot"),   # weather interacts with time of day
    ]

    global_mean = target.mean()

    for key in encode_keys:
        if isinstance(key, tuple):
            col_name = "_x_".join(key) + "_enc"
            train["_key"] = train[list(key)].astype(str).agg("_".join, axis=1)
            test["_key"]  = test[list(key)].astype(str).agg("_".join, axis=1)
        else:
            col_name = key + "_enc"
            train["_key"] = train[key].astype(str)
            test["_key"]  = test[key].astype(str)

        train[col_name] = np.nan

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train)):
            mapping = train.iloc[tr_idx].groupby("_key")["demand"].mean()
            train.loc[val_idx, col_name] = train.loc[val_idx, "_key"].map(mapping)

        # Full mapping for test
        full_map = train.groupby("_key")["demand"].mean()
        test[col_name] = test["_key"].map(full_map)

        # Fallback for unseen keys
        train[col_name].fillna(global_mean, inplace=True)
        test[col_name].fillna(global_mean, inplace=True)

    train.drop(columns=["_key"], inplace=True)
    test.drop(columns=["_key"],  inplace=True)

    return train, test


# ─────────────────────────────────────────────
# STEP 6 — AGGREGATE STATS
# ─────────────────────────────────────────────
def add_aggregate_stats(train, test):
    """Per-location and per-timeslot demand statistics."""
    print("\n[5/6] Aggregate statistics...")

    agg_configs = [
        ("geohash",        ["mean", "std", "median", "max", "min"]),
        ("time_slot",      ["mean", "std"]),
        ("geo_p4",         ["mean", "std"]),
        ("RoadType",       ["mean"]),
        ("NumberofLanes",  ["mean", "std"]),
        ("Weather",        ["mean"]),
    ]

    for key, aggs in agg_configs:
        stats = (train.groupby(key)["demand"]
                 .agg(aggs)
                 .add_prefix(f"{key}_")
                 .reset_index())
        train = train.merge(stats, on=key, how="left")
        test  = test.merge(stats,  on=key, how="left")

    # Derived ratio: how does this location's lag compare to its own baseline
    # Avoids division by zero with a small epsilon
    eps = 1e-6
    for df in [train, test]:
        df["lag_to_geo_mean_ratio"] = df["demand_d48"] / (df["geohash_mean"] + eps)
        df["lag_minus_geo_median"]  = df["demand_d48"] - df["geohash_median"]

        # Temperature bin (coarse: cold / mild / warm / hot)
        df["temp_bin"] = pd.cut(
            df["Temperature"],
            bins=[-999, 15, 22, 30, 999],
            labels=[0, 1, 2, 3]
        ).astype(float)

    # Fill any NaN from unseen values using train medians
    train_medians = train.select_dtypes(include=[np.number]).median()
    train = train.fillna(train_medians)
    test  = test.fillna(train_medians)

    return train, test


# ─────────────────────────────────────────────
# STEP 7 — ENCODE CATEGORICALS
# ─────────────────────────────────────────────
def encode_categoricals(train, test):
    cat_cols = ["geohash", "geo_p3", "geo_p4", "geo_p5", "geo_p6",
                "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    cat_cols = [c for c in cat_cols if c in train.columns]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        encoders[col] = le

    return train, test, encoders


# ─────────────────────────────────────────────
# STEP 8 — TRAIN + PREDICT
# ─────────────────────────────────────────────
def train_and_predict(train, test, kf):
    print("\n[6/6] Training LightGBM (5-fold CV)...")

    drop_cols = ["Index", "demand", "timestamp", "hour", "minute"]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X      = train[feature_cols]
    y      = train["demand"]
    X_test = test[feature_cols]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}\n")

    oof_preds   = np.zeros(len(train))
    test_preds  = np.zeros(len(test))
    models      = []
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(500),
            ],
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / N_FOLDS

        fold_r2    = r2_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_score = max(0, 100 * fold_r2)
        fold_scores.append(fold_score)
        models.append(model)

        print(f"  Fold {fold+1} | Score: {fold_score:.4f} | "
              f"Trees: {model.best_iteration_} | "
              f"Time: {time.time()-t0:.1f}s")

    oof_score = max(0, 100 * r2_score(y, oof_preds))
    print(f"\n  ── OOF Score: {oof_score:.4f} ──")
    print(f"  ── Fold Scores: {[f'{s:.2f}' for s in fold_scores]} ──")

    return models, oof_preds, test_preds, feature_cols, oof_score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction")
    print("=" * 55)

    train_raw, train, test = load_data()

    print("\n[2/6] Feature engineering...")
    train_raw = add_temporal_features(train_raw)
    train_raw = add_geo_features(train_raw)
    train = add_temporal_features(train)
    test  = add_temporal_features(test)
    train = add_geo_features(train)
    test  = add_geo_features(test)
    train, test = fill_missing(train, test, train_raw=train_raw)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train, test = add_lag_features(train_raw, train, test)
    train, test = add_target_encoding(train, test, kf)
    train, test = add_aggregate_stats(train, test)
    train, test, encoders = encode_categoricals(train, test)

    models, oof_preds, test_preds, feature_cols, oof_score = train_and_predict(train, test, kf)

    # ── Save models
    save_path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols,
                     "encoders": encoders, "oof_score": oof_score}, f)
    print(f"\n  Models saved → {save_path}")

    # ── Save submission
    submission = pd.DataFrame({
        "Index":  test["Index"],
        "demand": test_preds,
    })
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved → {sub_path}  shape: {submission.shape}")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Final OOF Score: {oof_score:.4f} / 100")
    print("=" * 55)


if __name__ == "__main__":
    main()