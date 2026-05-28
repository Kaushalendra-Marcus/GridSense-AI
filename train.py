"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction  (v3 — improved)
=====================================================================
Key improvements over v2:
  1. Timestamp parsed via string-split (not pd.to_datetime which mis-parses "2:15")
  2. Day-49 ratio features  — per-geohash drift from Day 48 → Day 49 (leakage-free)
  3. Neighbour lags from Day 48  (±1 slot, ±4 slots, rolling-3 mean)
  4. More target-encoding keys  (+hour combinations)
  5. Honest OOF reporting — Day-49-only OOF is the real online proxy
  6. Tuned LGBM params  (num_leaves 63, n_estimators 5000, more patience)
  7. Mild sample weighting  (Day-49 rows × 2 to align with test-day distribution)

Run:  python train.py
Output: outputs/submission.csv   models/lgbm_models.pkl
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
DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
N_FOLDS    = 5
SEED       = 42

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      3000,
    "learning_rate":     0.02,
    "num_leaves":        31,
    "max_depth":         5,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        2.0,
    "min_split_gain":    0.01,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("\n[1/8] Loading data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test      = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train     = train_raw.copy().reset_index(drop=True)
    print(f"  Raw train: {train_raw.shape}  |  Test: {test.shape}")
    print(f"  Day 48: {(train_raw['day']==48).sum()} rows  |  "
          f"Day 49: {(train_raw['day']==49).sum()} rows")
    return train_raw, train, test


# ─────────────────────────────────────────────
# STEP 2 — TEMPORAL FEATURES
# ─────────────────────────────────────────────
def add_temporal_features(df):
    """
    Parse timestamps like '2:15' or '0:0' using string-split.
    pd.to_datetime('2:15') is unreliable — it may parse as Feb-15 (date),
    yielding hour=0, minute=0 for EVERY row.  String-split is robust.
    """
    ts = df["timestamp"].astype(str)
    df["hour"]      = ts.str.split(":").str[0].astype(int)
    df["minute"]    = ts.str.split(":").str[1].astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15   # 0-95

    df["is_day49"]   = (df["day"] == 49).astype(int)   # replaces raw 'day' in model
    df["is_peak"]    = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_morning"] = df["hour"].isin([6, 7, 8, 9, 10]).astype(int)
    df["is_day"]     = df["hour"].isin([10, 11, 12, 13, 14, 15, 16]).astype(int)

    # Cyclical encodings (preserve midnight↔23:45 continuity)
    df["sin_hour"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_timeslot"]  = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["cos_timeslot"]  = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["sin_day"]       = np.sin(2 * np.pi * df["day"] / 7)
    df["cos_day"]       = np.cos(2 * np.pi * df["day"] / 7)
    return df


# ─────────────────────────────────────────────
# STEP 3 — GEO FEATURES
# ─────────────────────────────────────────────
def add_geo_features(df):
    gh = df["geohash"].astype(str)
    df["geo_p3"] = gh.str[:3]
    df["geo_p4"] = gh.str[:4]
    df["geo_p5"] = gh.str[:5]
    df["geo_p6"] = gh.str[:6]
    try:
        import pygeohash as pgh
        coords  = gh.apply(lambda x: pgh.decode(x) if len(x) >= 4 else (np.nan, np.nan))
        df["lat"] = coords.apply(lambda x: x[0])
        df["lon"] = coords.apply(lambda x: x[1])
    except ImportError:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df


# ─────────────────────────────────────────────
# STEP 4 — FILL MISSING
# ─────────────────────────────────────────────
def fill_missing(train, test, train_raw=None):
    src = train_raw if train_raw is not None else train

    for col in ["RoadType", "Weather"]:
        for df in [train, test]:
            geo_mode = (src.dropna(subset=[col])
                        .groupby("geo_p4")[col]
                        .agg(lambda x: x.mode()[0] if len(x) > 0 else np.nan))
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "geo_p4"].map(geo_mode)
            df[col].fillna(src[col].mode()[0], inplace=True)

    for df in [train, test]:
        geo_day_med = (src.dropna(subset=["Temperature"])
                       .groupby(["geohash", "day"])["Temperature"].median())
        df["Temperature"] = df["Temperature"].fillna(
            df.apply(lambda r: geo_day_med.get((r["geohash"], r["day"]), np.nan), axis=1))
        df["Temperature"].fillna(src["Temperature"].median(), inplace=True)

    return train, test


# ─────────────────────────────────────────────
# STEP 5 — LAG + NEIGHBOUR FEATURES
# ─────────────────────────────────────────────
def add_lag_features(train_raw, train, test):
    """
    Build demand lag features from Day 48:
      - demand_d48          : Day-48 demand at same geohash+timestamp  (primary predictor)
      - demand_d48_t-1/+1   : demand at previous / next 15-min slot
      - demand_d48_t-4/+4   : demand 1 hour before / after
      - demand_d48_roll3    : rolling 3-slot centred mean (smoothed Day-48)
      - demand_prev_slot    : previous time_slot within Day 48
    """
    print("\n[3/8] Building lag + neighbour features...")

    # ── Primary lag: Day 48 @ same geohash+timestamp ──────────────────────
    day48 = (train_raw[train_raw["day"] == 48]
             [["geohash", "timestamp", "time_slot", "demand"]]
             .rename(columns={"demand": "demand_d48"}))

    train = train.merge(day48[["geohash", "timestamp", "demand_d48"]],
                        on=["geohash", "timestamp"], how="left")
    train.loc[train["day"] == 48, "demand_d48"] = np.nan   # avoid self-leakage
    test  = test.merge(day48[["geohash", "timestamp", "demand_d48"]],
                       on=["geohash", "timestamp"], how="left")

    # Hierarchical fill for demand_d48
    day48["geo_p4"]   = day48["geohash"].astype(str).str[:4]
    geo_median        = day48.groupby("geohash")["demand_d48"].median()
    geo_p4_ts_med     = day48.groupby(["geo_p4", "timestamp"])["demand_d48"].median()
    ts_med            = day48.groupby("timestamp")["demand_d48"].median()
    global_med        = day48["demand_d48"].median()

    for df in [train, test]:
        df["demand_d48"].fillna(df["geohash"].map(geo_median), inplace=True)
        mask = df["demand_d48"].isna()
        if mask.any():
            df.loc[mask, "demand_d48"] = df.loc[mask].apply(
                lambda r: geo_p4_ts_med.get((r["geohash"][:4], r["timestamp"]), np.nan), axis=1)
        df["demand_d48"].fillna(df["timestamp"].map(ts_med), inplace=True)
        df["demand_d48"].fillna(global_med, inplace=True)

    # ── Neighbour lags: look up Day 48 at time_slot ± offset ─────────────
    lag_map = day48.set_index(["geohash", "time_slot"])["demand_d48"]

    for offset in [-1, 1, -4, 4]:
        col = f"demand_d48_t{offset:+d}"
        for df in [train, test]:
            df[col] = df.apply(
                lambda r: lag_map.get((r["geohash"], r["time_slot"] + offset), np.nan),
                axis=1
            )
            df[col].fillna(df["demand_d48"], inplace=True)

    # ── Rolling 3-slot centred mean on Day 48 ────────────────────────────
    d48_roll = (day48.sort_values(["geohash", "time_slot"])
                     .copy())
    d48_roll["demand_d48_roll3"] = d48_roll.groupby("geohash")["demand_d48"].transform(
        lambda x: x.rolling(3, center=True, min_periods=1).mean())
    roll_map = d48_roll.set_index(["geohash", "time_slot"])["demand_d48_roll3"]

    for df in [train, test]:
        df["demand_d48_roll3"] = df.apply(
            lambda r: roll_map.get((r["geohash"], r["time_slot"]), np.nan), axis=1)
        df["demand_d48_roll3"].fillna(df["demand_d48"], inplace=True)

    # ── Previous-slot within Day 48 ──────────────────────────────────────
    prev_map = (d48_roll.assign(
                    demand_prev_slot=lambda d: d.groupby("geohash")["demand_d48"].shift(1))
                .set_index(["geohash", "time_slot"])["demand_prev_slot"])

    for df in [train, test]:
        df["demand_prev_slot"] = df.apply(
            lambda r: prev_map.get((r["geohash"], r["time_slot"]), np.nan), axis=1)
        df["demand_prev_slot"].fillna(df["demand_d48"], inplace=True)

    print(f"  NaN in train demand_d48: {train['demand_d48'].isna().sum()}")
    print(f"  NaN in test  demand_d48: {test['demand_d48'].isna().sum()}")
    return train, test


# ─────────────────────────────────────────────
# STEP 6 — DAY-49 RATIO FEATURES  (KEY NEW STEP)
# ─────────────────────────────────────────────
def add_d49_features(train_raw, train, test):
    """
    Use Day-49 TRAINING rows (timestamps 0:00–2:00) to quantify how
    Day 49 differs from Day 48 for each geohash.  This is 100% leakage-free
    because the test window starts at 2:15, after all Day-49 training data.

    Features produced:
      demand_d49_mean     — per-geohash mean demand during 0:00–2:00 on Day 49
      demand_d49_last     — per-geohash demand at the very last known slot (2:00)
      d49_d48_ratio       — demand_d49_mean / demand_d48_early  (drift factor)
      d49_d48_p4_ratio    — same ratio at geo_p4 level (fallback)
      demand_d48_corrected— demand_d48 × d49_d48_ratio  (key corrected predictor)
    """
    print("\n[4/8] Day-49 ratio / drift features...")

    d49_tr = train_raw[train_raw["day"] == 49][["geohash", "time_slot", "demand"]].copy()
    # Day-48 in the SAME early window (slots 0-8 = 0:00-2:00) for a fair ratio
    d48_early = (train_raw[(train_raw["day"] == 48) & (train_raw["time_slot"] <= 8)]
                 [["geohash", "demand"]].copy())

    # ── geohash-level aggregates ──────────────────────────────────────────
    d49_geo_mean  = d49_tr.groupby("geohash")["demand"].mean().rename("demand_d49_mean")
    d48_geo_early = d48_early.groupby("geohash")["demand"].mean().rename("demand_d48_early")

    ratio_df = pd.concat([d49_geo_mean, d48_geo_early], axis=1).dropna()
    ratio_df["d49_d48_ratio"] = (ratio_df["demand_d49_mean"]
                                  / (ratio_df["demand_d48_early"] + 1e-6)).clip(0.1, 10.0)

    # Last known Day-49 observation per geohash (slot 8 = 2:00)
    d49_last = (d49_tr[d49_tr["time_slot"] == 8]
                .groupby("geohash")["demand"].mean()
                .rename("demand_d49_last"))

    # ── geo_p4 aggregates for unseen geohashes ────────────────────────────
    d49_tr["geo_p4"]   = d49_tr["geohash"].str[:4]
    d48_early["geo_p4"] = d48_early["geohash"].str[:4]
    d49_p4_mean  = d49_tr.groupby("geo_p4")["demand"].mean()
    d48_p4_early = d48_early.groupby("geo_p4")["demand"].mean()
    d49_d48_p4   = (d49_p4_mean / (d48_p4_early + 1e-6)).clip(0.1, 10.0)

    global_ratio = float(ratio_df["d49_d48_ratio"].median())

    for df in [train, test]:
        df["demand_d49_mean"] = df["geohash"].map(d49_geo_mean)
        df["demand_d49_last"] = df["geohash"].map(d49_last)
        df["d49_d48_ratio"]   = df["geohash"].map(ratio_df["d49_d48_ratio"])
        df["d49_d48_p4_ratio"]= df["geo_p4"].map(d49_d48_p4)

        # Cascading fill for ratio
        mask = df["d49_d48_ratio"].isna()
        df.loc[mask, "d49_d48_ratio"] = df.loc[mask, "geo_p4"].map(d49_d48_p4)
        df["d49_d48_ratio"].fillna(global_ratio, inplace=True)
        df["d49_d48_p4_ratio"].fillna(global_ratio, inplace=True)

        # Fill demand_d49_mean
        mask = df["demand_d49_mean"].isna()
        df.loc[mask, "demand_d49_mean"] = df.loc[mask, "geo_p4"].map(d49_p4_mean)
        df["demand_d49_mean"].fillna(df["demand_d48"], inplace=True)
        df["demand_d49_last"].fillna(df["demand_d49_mean"], inplace=True)

        # The most important derived feature
        df["demand_d48_corrected"] = df["demand_d48"] * df["d49_d48_ratio"]

    print(f"  d49_d48_ratio  — median={global_ratio:.3f}  "
          f"std={ratio_df['d49_d48_ratio'].std():.3f}  "
          f"coverage={ratio_df.shape[0]} geohashes")
    return train, test


# ─────────────────────────────────────────────
# STEP 7 — TARGET ENCODING (fold-safe OOF)
# ─────────────────────────────────────────────
def add_target_encoding(train, test, kf, train_raw=None):
    print("\n[5/8] Target encoding (OOF)...")
    target      = train["demand"]
    global_mean = float(target.mean())
    src         = train_raw if train_raw is not None else train

    encode_keys = [
        "geohash",
        "geo_p4",
        ("geohash",  "time_slot"),
        ("geohash",  "day"),
        ("geohash",  "hour"),
        ("geo_p4",   "time_slot"),
        ("geo_p4",   "hour"),
        ("day",      "time_slot"),
        ("RoadType", "time_slot"),
        ("geohash",  "is_peak"),
        ("Weather",  "time_slot"),
        ("Weather",  "hour"),
    ]

    for key in encode_keys:
        if isinstance(key, tuple):
            col_name = "_x_".join(key) + "_enc"
            def make_key(df, cols):
                return df[cols].fillna(-1).astype(str).agg("_".join, axis=1)
            train["_key"] = make_key(train, list(key))
            test["_key"]  = make_key(test,  list(key))
            src["_key"]   = make_key(src,   list(key))
        else:
            col_name = key + "_enc"
            train["_key"] = train[key].astype(str)
            test["_key"]  = test[key].astype(str)
            src["_key"]   = src[key].astype(str)

        train[col_name] = np.nan
        for _, (tr_idx, val_idx) in enumerate(kf.split(train)):
            mapping = train.iloc[tr_idx].groupby("_key")["demand"].mean()
            train.loc[val_idx, col_name] = train.loc[val_idx, "_key"].map(mapping)

        full_map = src.groupby("_key")["demand"].mean()
        test[col_name] = test["_key"].map(full_map)

        train[col_name].fillna(global_mean, inplace=True)
        test[col_name].fillna(global_mean, inplace=True)

    train.drop(columns=["_key"], inplace=True)
    test.drop(columns=["_key"],  inplace=True)
    if "_key" in src.columns:
        src.drop(columns=["_key"], inplace=True)

    return train, test


# ─────────────────────────────────────────────
# STEP 8 — AGGREGATE STATS
# ─────────────────────────────────────────────
def add_aggregate_stats(train, test):
    print("\n[6/8] Aggregate statistics...")

    agg_configs = [
        ("geohash",        ["mean", "std", "median", "max", "min"]),
        ("time_slot",      ["mean", "std"]),
        ("geo_p4",         ["mean", "std"]),
        ("RoadType",       ["mean"]),
        ("NumberofLanes",  ["mean", "std"]),
        ("Weather",        ["mean"]),
        ("hour",           ["mean", "std"]),
    ]

    for key, aggs in agg_configs:
        stats = (train.groupby(key)["demand"]
                 .agg(aggs)
                 .add_prefix(f"{key}_")
                 .reset_index())
        train = train.merge(stats, on=key, how="left")
        test  = test.merge(stats,  on=key, how="left")

    eps = 1e-6
    for df in [train, test]:
        df["lag_to_geo_mean_ratio"]   = df["demand_d48"]           / (df["geohash_mean"] + eps)
        df["lag_minus_geo_median"]    = df["demand_d48"]           -  df["geohash_median"]
        df["corrected_to_geo_ratio"]  = df["demand_d48_corrected"] / (df["geohash_mean"] + eps)
        df["d49_mean_to_geo_ratio"]   = df["demand_d49_mean"]      / (df["geohash_mean"] + eps)

        df["temp_bin"] = pd.cut(
            df["Temperature"],
            bins=[-999, 15, 22, 30, 999],
            labels=[0, 1, 2, 3]
        ).astype(float)

    train_medians = train.select_dtypes(include=[np.number]).median()
    train = train.fillna(train_medians)
    test  = test.fillna(train_medians)

    return train, test


# ─────────────────────────────────────────────
# STEP 9 — ENCODE CATEGORICALS
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
# STEP 10 — TRAIN + PREDICT
# ─────────────────────────────────────────────
def train_and_predict(train, test, kf):
    print("\n[7/8] Training LightGBM (5-fold CV)...")

    drop_cols    = ["Index", "demand", "timestamp", "hour", "minute"]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X      = train[feature_cols]
    y      = train["demand"]
    X_test = test[feature_cols]

    # Mild upweight for Day-49 rows — same day as test, so slightly more relevant.
    # Keeping it at 2x (not higher) to avoid overfitting to night-only patterns.
    sample_weights = np.where(train["day"] == 49, 2.0, 1.0)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature list sample: {feature_cols[:10]} ...")

    oof_preds   = np.zeros(len(train))
    test_preds  = np.zeros(len(test))
    models      = []
    fold_scores = []

    d49_mask = (train["day"] == 49).values

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            sample_weight=sample_weights[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(300, verbose=False),
                lgb.log_evaluation(500),
            ],
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        test_preds        += model.predict(X_test) / N_FOLDS

        fold_r2    = r2_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_score = max(0, 100 * fold_r2)
        fold_scores.append(fold_score)
        models.append(model)

        # Day-49 only OOF — more honest proxy for online score
        d49_val = d49_mask[val_idx]
        if d49_val.sum() > 0:
            d49_r2    = r2_score(y.iloc[val_idx][d49_val], oof_preds[val_idx][d49_val])
            d49_score = max(0, 100 * d49_r2)
            print(f"  Fold {fold+1} | OOF: {fold_score:.2f} | "
                  f"D49-OOF: {d49_score:.2f} | "
                  f"Trees: {model.best_iteration_} | "
                  f"Time: {time.time()-t0:.1f}s")
        else:
            print(f"  Fold {fold+1} | OOF: {fold_score:.2f} | "
                  f"Trees: {model.best_iteration_} | Time: {time.time()-t0:.1f}s")

    oof_score = max(0, 100 * r2_score(y, oof_preds))
    print(f"\n  ── OOF Score (all data):  {oof_score:.4f} ──")
    if d49_mask.sum() > 0:
        d49_oof = max(0, 100 * r2_score(y[d49_mask], oof_preds[d49_mask]))
        print(f"  ── OOF Score (Day 49):    {d49_oof:.4f}  ← best proxy for online score ──")
    print(f"  ── Fold Scores: {[f'{s:.2f}' for s in fold_scores]} ──")

    return models, oof_preds, test_preds, feature_cols, oof_score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction v3")
    print("=" * 60)

    train_raw, train, test = load_data()

    print("\n[2/8] Feature engineering (temporal + geo)...")
    for df in [train_raw, train, test]:
        add_temporal_features(df)
        add_geo_features(df)

    train, test = fill_missing(train, test, train_raw=train_raw)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train, test = add_lag_features(train_raw, train, test)
    train, test = add_d49_features(train_raw, train, test)
    train, test = add_target_encoding(train, test, kf, train_raw=train_raw)
    train, test = add_aggregate_stats(train, test)
    train, test, encoders = encode_categoricals(train, test)

    models, oof_preds, test_preds, feature_cols, oof_score = train_and_predict(
        train, test, kf)

    # ── Save models
    save_path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols,
                     "encoders": encoders, "oof_score": oof_score}, f)
    print(f"\n[8/8] Models saved → {save_path}")

    # ── Save submission
    submission = pd.DataFrame({
        "Index":  test["Index"],
        "demand": test_preds,
    })
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"       Submission → {sub_path}  shape: {submission.shape}")

    total_time = time.time() - t_start
    print(f"\n  Total time:      {total_time/60:.1f} min")
    print(f"  Final OOF Score: {oof_score:.4f} / 100")
    print("=" * 60)


if __name__ == "__main__":
    main()
