"""
Gridlock Hackathon 2.0 — Traffic Demand Prediction
====================================================
v4 — Rebuilt from EDA insights:

  KEY DATA FACTS (from eda.py analysis):
  - D49-train : slots 0-8  (0:00-2:00)   9 slots, ~870 geohashes
  - Test       : slots 9-55 (2:15-13:45) 47 slots, ~480 geohashes
  - D48        : all 96 slots, ~723 geohashes  (covers the full test window)
  - Zero timestamp overlap between D49-train and test -> d49_d48_ratio is leak-free
  - Lag coverage per test geohash ranges 0-100% (NOT uniformly 100%)
  - Drift (D49/D48) is rapidly declining: 2.30 at 0:00 -> 1.29 at 2:00
  - Missing values are geohash-structural, not random

  v4 DESIGN DECISIONS:
  1. Train on D48 + D49 (sample_weight=4 for D49 rows)
     D48 covers test time slots so the model learns daytime patterns.
  2. Lag for D48 training rows = geohash x slot median of D48 itself
     (mild circular reference, controlled by is_lag_real=0 flag)
  3. Lag for D49-train and test = exact D48 match + fallback chain
  4. d49_d48_ratio clipped to [0.5, 3.0] and weighted by slot recency
     (later slots in D49-train window get higher weight)
  5. Aggregate stats sourced from D48 only -> zero leakage in D49 OOF
  6. Missing values filled geohash-first, then global
  7. OOF metric reported separately for D48 rows vs D49 rows;
     D49-only OOF is the honest proxy for online score

Run:  python train.py
Out:  outputs/submission.csv  |  models/lgbm_models.pkl
"""

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
SEED       = 42
N_FOLDS    = 5

# Validation/training controls for robustness against public-test overfitting.
CV_MODE = "group_geohash"  # options: "group_geohash", "random"
DROP_SUSPECT_FEATURES_FOR_TRAINING = False

SUSPECT_FEATURES = {
    "demand_d48",
    "demand_d48_corrected",
    "gh_ts_d48_mean",
    "gh_ts_d48_std",
    "p4_ts_d48_mean",
}

np.random.seed(SEED)

# D49 rows get 4x weight -> emphasise Day-49 patterns since test is Day 49
D49_SAMPLE_WEIGHT = 4.0

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      8000,
    "learning_rate":     0.01,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1/8] Loading data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test      = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    d48 = train_raw[train_raw["day"] == 48].copy().reset_index(drop=True)
    d49 = train_raw[train_raw["day"] == 49].copy().reset_index(drop=True)

    print(f"  D48 train : {len(d48)} rows  |  D49 train : {len(d49)} rows  |  Test : {len(test)} rows")
    print(f"  D48 geohashes: {d48['geohash'].nunique()}  |  D49 geohashes: {d49['geohash'].nunique()}  |  Test geohashes: {test['geohash'].nunique()}")
    return train_raw, d48, d49, test


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────

def add_temporal_features(df):
    ts = df["timestamp"].astype(str)
    df = df.copy()
    df["hour"]      = ts.str.split(":").str[0].astype(int)
    df["minute"]    = ts.str.split(":").str[1].astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15

    # Flags
    df["is_peak"]  = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_day49"] = (df["day"] == 49).astype(int)

    # Cyclical encodings
    df["sin_slot"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["cos_slot"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"]      / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"]      / 24)

    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — GEO FEATURES
# ─────────────────────────────────────────────────────────────────────

def add_geo_features(df):
    gh = df["geohash"].astype(str)
    df = df.copy()
    df["geo_p3"] = gh.str[:3]
    df["geo_p4"] = gh.str[:4]
    df["geo_p5"] = gh.str[:5]
    df["geo_p6"] = gh.str[:6]
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — MISSING VALUE FILL
# EDA finding: missing is geohash-structural, not random.
# Fill order: geohash mode -> geo_p4 mode -> global mode/median
# ─────────────────────────────────────────────────────────────────────

def fill_missing(train, test, d48_src):
    """Fill missing values using D48 as the reference distribution."""
    train = train.copy()
    test  = test.copy()

    # --- Categorical: RoadType, Weather ---
    for col in ["RoadType", "Weather"]:
        # Compute geohash-level mode from D48
        gh_mode = (d48_src.dropna(subset=[col])
                   .groupby("geohash")[col]
                   .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan))
        p4_mode = (d48_src.dropna(subset=[col])
                   .groupby("geo_p4")[col]
                   .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan))
        global_mode = d48_src[col].mode().iloc[0]

        for df in [train, test]:
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "geohash"].map(gh_mode)
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "geo_p4"].map(p4_mode)
            df[col] = df[col].fillna(global_mode)

    # --- Temperature ---
    gh_temp_med = d48_src.dropna(subset=["Temperature"]).groupby("geohash")["Temperature"].median()
    global_temp = d48_src["Temperature"].median()
    for df in [train, test]:
        mask = df["Temperature"].isna()
        df.loc[mask, "Temperature"] = df.loc[mask, "geohash"].map(gh_temp_med)
        df["Temperature"] = df["Temperature"].fillna(global_temp)

    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — LAG FEATURES
#
# D48 training rows: no Day-47 data exists.
#   -> lag = geohash x time_slot median from D48 itself (circular but mild)
#   -> is_lag_real = 0
#
# D49 training rows and test rows: exact D48 match by geohash+timestamp.
#   -> is_lag_real = 1 if matched, 0 after fallback
# ─────────────────────────────────────────────────────────────────────

def add_lag_features(d48, d49_train, test):
    print("\n[3/8] Building lag features...")

    # ── D48 self-lag (proxy lag for D48 training rows) ────────────────
    # Geohash x slot median EXCLUDING the row itself is ideal but expensive.
    # We use the full-group median as an approximation; the model discounts
    # this via is_lag_real=0.
    gh_slot_med_d48 = (d48.groupby(["geohash", "time_slot"])["demand"]
                       .median()
                       .rename("demand_d48"))
    # Fallback levels for proxy lag
    slot_med_d48    = d48.groupby("time_slot")["demand"].median()
    gh_med_d48      = d48.groupby("geohash")["demand"].median()

    d48 = d48.copy()
    d48["demand_d48"] = d48.set_index(["geohash", "time_slot"]).index.map(gh_slot_med_d48)
    mask = d48["demand_d48"].isna()
    d48.loc[mask, "demand_d48"] = d48.loc[mask, "time_slot"].map(slot_med_d48)
    d48["demand_d48"] = d48["demand_d48"].fillna(d48["demand"].median())
    d48["is_lag_real"] = 0  # proxy, not a real previous-day observation

    # ── Real lag for D49-train and test (D48 demand at same location+time) ──
    d48_exact = d48.set_index(["geohash", "timestamp"])["demand"]

    # Fallback chain: geohash median -> geo_p4+timestamp median -> timestamp median -> global
    gh_ts_med  = (d48.groupby(["geohash",  "timestamp"])["demand"].median())
    p4_ts_med  = (d48.groupby(["geo_p4",   "timestamp"])["demand"].median())
    ts_med     = d48.groupby("timestamp")["demand"].median()
    global_med = d48["demand"].median()

    def attach_lag(df):
        df = df.copy()
        # Exact match
        df["demand_d48"] = [d48_exact.get((g, t), np.nan)
                            for g, t in zip(df["geohash"], df["timestamp"])]
        df["is_lag_real"] = df["demand_d48"].notna().astype(int)

        # Fallback 1: geohash median
        m = df["demand_d48"].isna()
        df.loc[m, "demand_d48"] = df.loc[m, "geohash"].map(gh_med_d48)

        # Fallback 2: geo_p4 + timestamp median
        m = df["demand_d48"].isna()
        if m.any():
            df.loc[m, "demand_d48"] = df.loc[m].apply(
                lambda r: p4_ts_med.get((r["geo_p4"], r["timestamp"]), np.nan), axis=1)

        # Fallback 3: timestamp median
        m = df["demand_d48"].isna()
        df.loc[m, "demand_d48"] = df.loc[m, "timestamp"].map(ts_med)

        # Fallback 4: global median
        df["demand_d48"] = df["demand_d48"].fillna(global_med)
        return df

    d49_train = attach_lag(d49_train)
    test      = attach_lag(test)

    cov_d49 = (d49_train["is_lag_real"]).mean()
    cov_te  = (test["is_lag_real"]).mean()
    print(f"  D49-train exact lag coverage : {cov_d49*100:.1f}%")
    print(f"  Test     exact lag coverage  : {cov_te*100:.1f}%")

    return d48, d49_train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 6 — D49 DRIFT FEATURES
#
# EDA finding: drift is rapidly declining (2.30 at 0:00 -> 1.29 at 2:00).
# We compute per-geohash ratio with slot-recency weighting so later slots
# (closer to test window) contribute more. Clip to [0.5, 3.0] to suppress
# noisy ratios from 1-2 sample geohashes.
# This is 100% leak-free for test (non-overlapping time windows).
# For D49-OOF: computed from training-fold D49 rows only (OOF-safe).
# ─────────────────────────────────────────────────────────────────────

def compute_d49_ratio(d49_rows, d48_src, clip=(0.5, 3.0)):
    """
    Compute per-geohash D49/D48 drift ratio from a subset of D49 rows.
    Slots closer to the test window (slot 8 > slot 0) get higher weight.
    Returns a Series indexed by geohash.
    """
    d48_map   = d48_src.set_index(["geohash", "timestamp"])["demand"]
    d49r      = d49_rows.copy()
    d49r["demand_d48_real"] = [d48_map.get((g, t), np.nan)
                               for g, t in zip(d49r["geohash"], d49r["timestamp"])]
    matched = d49r.dropna(subset=["demand_d48_real"])

    if len(matched) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, 1.0

    # Slot-recency weight: slot 8 gets weight 8, slot 0 gets weight 1 (linear)
    matched = matched.copy()
    matched["slot_weight"] = matched["time_slot"] + 1   # slots 0-8 -> weights 1-9

    def geo_ratio(grp):
        w       = grp["slot_weight"].values
        d49_w   = np.average(grp["demand"].values,      weights=w)
        d48_w   = np.average(grp["demand_d48_real"].values, weights=w)
        if d48_w < 1e-6:
            return np.nan
        return d49_w / d48_w

    ratios = matched.groupby("geohash").apply(geo_ratio)
    ratios = ratios.clip(*clip)

    # Fallback: geo_p4 level ratio
    matched["geo_p4"] = matched["geohash"].str[:4]
    p4_ratios = matched.groupby("geo_p4").apply(geo_ratio).clip(*clip)
    global_ratio = np.nanmedian(ratios.values)
    if not np.isfinite(global_ratio):
        global_ratio = 1.0

    return ratios, p4_ratios, global_ratio


def attach_d49_features(df, ratios, p4_ratios, global_ratio):
    df = df.copy()
    gh_ratio = df["geohash"].map(ratios)
    mask     = gh_ratio.isna()
    if mask.any():
        gh_ratio[mask] = df.loc[mask, "geo_p4"].map(p4_ratios)
    gh_ratio = gh_ratio.fillna(global_ratio)

    df["d49_d48_ratio"]        = gh_ratio
    df["demand_d48_corrected"] = df["demand_d48"] * gh_ratio
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 7 — AGGREGATE STATS
#
# EDA finding: use D48 only as source -> zero leakage into D49 OOF folds.
# D48 covers all 96 slots including test slots 9-55.
# ─────────────────────────────────────────────────────────────────────

def add_aggregate_stats(train, test, d48_src):
    """Group demand statistics sourced exclusively from D48."""
    print("\n[5/8] Aggregate statistics (source: D48 only)...")

    agg_configs = [
        ("geohash",       ["mean", "std", "median", "max", "min"]),
        ("time_slot",     ["mean", "std"]),
        ("geo_p4",        ["mean", "std"]),
        ("geo_p5",        ["mean"]),
        ("RoadType",      ["mean"]),
        ("NumberofLanes", ["mean"]),
        ("Weather",       ["mean"]),
    ]

    for key, aggs in agg_configs:
        stats = (d48_src.groupby(key)["demand"]
                 .agg(aggs)
                 .add_prefix(f"{key}_d48_")
                 .reset_index())
        train = train.merge(stats, on=key, how="left")
        test  = test.merge(stats,  on=key, how="left")

    # Also add geohash x test_slot (slot 9-55) stats from D48
    # This is the most direct signal for the test window
    test_slots_d48 = d48_src[d48_src["time_slot"].between(9, 55)]
    if len(test_slots_d48) > 0:
        gh_ts_stats = (test_slots_d48.groupby(["geohash", "time_slot"])["demand"]
                       .agg(["mean", "std"])
                       .reset_index()
                       .rename(columns={"mean": "gh_ts_d48_mean",
                                        "std":  "gh_ts_d48_std"}))
        train = train.merge(gh_ts_stats, on=["geohash", "time_slot"], how="left")
        test  = test.merge(gh_ts_stats,  on=["geohash", "time_slot"], how="left")

        p4_ts_stats = (test_slots_d48.groupby(["geo_p4", "time_slot"])["demand"]
                       .mean()
                       .reset_index()
                       .rename(columns={"demand": "p4_ts_d48_mean"}))
        train = train.merge(p4_ts_stats, on=["geo_p4", "time_slot"], how="left")
        test  = test.merge(p4_ts_stats,  on=["geo_p4", "time_slot"], how="left")

    eps = 1e-6
    for df in [train, test]:
        df["lag_to_geo_mean_ratio"]  = df["demand_d48"] / (df["geohash_d48_mean"] + eps)
        df["lag_minus_geo_median"]   = df["demand_d48"] - df["geohash_d48_median"]
        df["lag_to_slot_mean_ratio"] = df["demand_d48"] / (df["time_slot_d48_mean"] + eps)
        df["temp_bin"] = pd.cut(df["Temperature"],
                                bins=[-999, 15, 22, 30, 999],
                                labels=[0, 1, 2, 3]).astype(float)

    # Fill NaN stats with training medians
    num_cols      = train.select_dtypes(include=[np.number]).columns
    train_medians = train[num_cols].median()
    train         = train.fillna(train_medians)
    test          = test.fillna(train_medians)

    return train, test


def build_cv_splits(train, n_folds=N_FOLDS, mode=CV_MODE):
    """Build reusable CV splits so all OOF stages use the exact same folds."""
    if mode == "group_geohash":
        groups = train["geohash"].astype(str)
        splitter = GroupKFold(n_splits=n_folds)
        splits = list(splitter.split(train, groups=groups))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(splitter.split(train))

    return splits


# ─────────────────────────────────────────────────────────────────────
# STEP 8 — TARGET ENCODING (OOF-safe)
# ─────────────────────────────────────────────────────────────────────

def add_target_encoding(train, test, cv_splits, d48_src):
    """
    OOF target encoding for train.
    Test uses the full training set (train + d48_src) for richer estimates.
    """
    print("\n[4/8] Target encoding (OOF-safe)...")
    global_mean = train["demand"].mean()
    # Full source for test mapping includes both days. Avoid duplicating
    # D48 rows (train already contains D48) which would skew encodings.
    full_src    = pd.concat([train, d48_src], ignore_index=True).drop_duplicates()

    encode_keys = [
        "geohash",
        "geo_p4",
        "geo_p5",
        ("geohash",  "time_slot"),
        ("geohash",  "is_day49"),
        ("geo_p4",   "time_slot"),
        ("geo_p5",   "time_slot"),
        ("day",      "time_slot"),
        ("RoadType", "time_slot"),
        ("Weather",  "time_slot"),
        ("geohash",  "is_peak"),
    ]

    for key in encode_keys:
        if isinstance(key, tuple):
            col_name     = "_x_".join(key) + "_enc"
            train["_k"] = train[list(key)].astype(str).agg("_".join, axis=1)
            test["_k"]  = test[list(key)].astype(str).agg("_".join, axis=1)
            full_src["_k"] = full_src[list(key)].astype(str).agg("_".join, axis=1)
        else:
            col_name     = key + "_enc"
            train["_k"] = train[key].astype(str)
            test["_k"]  = test[key].astype(str)
            full_src["_k"] = full_src[key].astype(str)

        train[col_name] = np.nan
        for _, (tr_idx, val_idx) in enumerate(cv_splits):
            mapping = train.iloc[tr_idx].groupby("_k")["demand"].mean()
            train.loc[val_idx, col_name] = train.loc[val_idx, "_k"].map(mapping)

        full_map        = full_src.groupby("_k")["demand"].mean()
        test[col_name]  = test["_k"].map(full_map)

        train[col_name] = train[col_name].fillna(global_mean)
        test[col_name]  = test[col_name].fillna(global_mean)

    train.drop(columns=["_k"], inplace=True)
    test.drop(columns=["_k"],  inplace=True)
    if "_k" in full_src.columns:
        full_src.drop(columns=["_k"], inplace=True)

    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 9 — LABEL ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# STEP 10 — TRAIN + PREDICT
# ─────────────────────────────────────────────────────────────────────

def train_and_predict(train, test, cv_splits, d48_src, encoders,
                      drop_suspect_features=DROP_SUSPECT_FEATURES_FOR_TRAINING):
    print("\n[6/8] Training LightGBM (5-fold CV)...")

    drop_cols = {"Index", "demand", "timestamp", "hour", "minute"}
    if drop_suspect_features:
        drop_cols |= SUSPECT_FEATURES

    feat_cols = [c for c in train.columns if c not in drop_cols]

    X         = train[feat_cols]
    y         = train["demand"]
    X_test    = test[feat_cols]

    # Sample weights: D49 rows get D49_SAMPLE_WEIGHT, D48 rows get 1.0
    weights = np.where(train["is_day49"].values == 1, D49_SAMPLE_WEIGHT, 1.0)

    print(f"  Features : {len(feat_cols)}")
    print(f"  Feature list: {feat_cols}\n")

    is_d49          = train["is_day49"].values == 1
    oof_preds       = np.zeros(len(train))
    test_preds      = np.zeros(len(test))
    models, scores  = [], []
    d49_oof_scores  = []

    n_splits = len(cv_splits)
    for fold, (tr_idx, val_idx) in enumerate(cv_splits):
        t0 = time.time()

        # OOF-safe d49_d48_ratio: recompute from training-fold D49 rows only
        d49_train_fold = train.iloc[tr_idx]
        d49_only_fold  = d49_train_fold[d49_train_fold["is_day49"] == 1]

        ratios_fold, p4_ratios_fold, global_ratio_fold = compute_d49_ratio(
            d49_only_fold.rename(columns={"demand_d48": "_lag_dummy"}
                         ).assign(demand=d49_only_fold["demand"]),
            d48_src)
        # For this fold: override ratio cols in validation rows
        X_tr  = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()

        # Ratios were computed on raw geohash strings. Our features are
        # label-encoded; transform ratio indexes to encoded labels if
        # encoders are available so mapping aligns correctly.
        ratios_fold_enc = ratios_fold.copy()
        p4_ratios_fold_enc = p4_ratios_fold.copy()
        if "geohash" in encoders and len(ratios_fold) > 0:
            try:
                enc_idx = encoders["geohash"].transform(ratios_fold.index.astype(str))
                ratios_fold_enc = pd.Series(ratios_fold.values, index=enc_idx)
            except Exception:
                # Fallback: keep original string-indexed ratios
                pass
        if "geo_p4" in encoders and len(p4_ratios_fold) > 0:
            try:
                enc_idx = encoders["geo_p4"].transform(p4_ratios_fold.index.astype(str))
                p4_ratios_fold_enc = pd.Series(p4_ratios_fold.values, index=enc_idx)
            except Exception:
                pass

        for df_part in [X_tr, X_val]:
            gh_ratio = df_part["geohash"].map(ratios_fold_enc)
            mask     = gh_ratio.isna()
            if mask.any():
                gh_ratio.loc[mask] = df_part.loc[mask, "geo_p4"].map(p4_ratios_fold_enc)
            gh_ratio = gh_ratio.fillna(global_ratio_fold)
            df_part["d49_d48_ratio"]        = gh_ratio.values
            df_part["demand_d48_corrected"] = (df_part["demand_d48"] * gh_ratio).values

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y.iloc[tr_idx],
            sample_weight=weights[tr_idx],
            eval_set=[(X_val, y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(400, verbose=False),
                lgb.log_evaluation(1000),
            ],
        )

        oof_preds[val_idx]  = model.predict(X_val)
        test_preds         += model.predict(X_test) / n_splits

        fold_r2    = r2_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_score = max(0, 100 * fold_r2)
        scores.append(fold_score)
        models.append(model)

        # D49-only OOF score (honest proxy for online score)
        val_is_d49  = is_d49[val_idx]
        if val_is_d49.sum() > 0:
            d49_r2 = r2_score(y.iloc[val_idx][val_is_d49],
                              oof_preds[val_idx][val_is_d49])
            d49_oof_scores.append(max(0, 100 * d49_r2))
            d49_str = f" | D49-OOF: {d49_oof_scores[-1]:.2f}"
        else:
            d49_str = ""

        print(f"  Fold {fold+1} | OOF: {fold_score:.2f}{d49_str} | "
              f"Trees: {model.best_iteration_} | Time: {time.time()-t0:.1f}s")

    oof_all = max(0, 100 * r2_score(y, oof_preds))
    oof_d49 = max(0, 100 * r2_score(y[is_d49], oof_preds[is_d49])) if is_d49.sum() > 0 else None

    print(f"\n  OOF Score (all data) : {oof_all:.4f}")
    if oof_d49 is not None:
        print(f"  OOF Score (D49 only) : {oof_d49:.4f}  <- honest proxy for online score")
    print(f"  Fold Scores          : {[f'{s:.2f}' for s in scores]}")

    return models, oof_preds, test_preds, feat_cols, oof_all, oof_d49


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 65)
    print("  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction  v4")
    print("=" * 65)

    # 1 — Load
    train_raw, d48, d49, test = load_data()

    # 2 — Temporal + geo features (apply to all splits)
    print("\n[2/8] Feature engineering (temporal + geo)...")
    for df_list in [[d48, d49, test]]:
        pass  # processed in place below

    d48  = add_temporal_features(d48)
    d49  = add_temporal_features(d49)
    test = add_temporal_features(test)
    d48  = add_geo_features(d48)
    d49  = add_geo_features(d49)
    test = add_geo_features(test)

    # d48_src used as reference for fill + stats
    d48_src = d48.copy()

    # 3 — Fill missing values (geohash-first strategy)
    # Combine D48+D49 as training data; apply fills to both + test
    train_combined = pd.concat([d48, d49], ignore_index=True)
    train_combined, test = fill_missing(train_combined, test, d48_src)
    # Re-split for lag building
    d48 = train_combined[train_combined["day"] == 48].copy()
    d49 = train_combined[train_combined["day"] == 49].copy()
    # Refresh d48_src after fill
    d48_src = d48.copy()

    # 4 — Lag features
    d48, d49, test = add_lag_features(d48, d49, test)

    # 5 — Combined training DataFrame
    train = pd.concat([d48, d49], ignore_index=True)

    # 6 — CV splits (group-based mode is stricter than random KFold)
    cv_splits = build_cv_splits(train, n_folds=N_FOLDS, mode=CV_MODE)
    print(f"\n[3a/8] CV mode: {CV_MODE} | folds: {len(cv_splits)}")

    # 7 — Pre-compute d49 ratio on FULL D49-train (for test predictions)
    print("\n[3b/8] Pre-computing D49 drift ratio (full D49-train)...")
    ratios_full, p4_ratios_full, global_ratio_full = compute_d49_ratio(d49, d48_src)
    print(f"  Global ratio (recency-weighted): {global_ratio_full:.4f}")
    print(f"  Geohashes with ratio data: {len(ratios_full)}")

    # Attach drift features to train and test using FULL ratio
    # (fold-level ratios used inside train_and_predict for OOF)
    train = attach_d49_features(train, ratios_full, p4_ratios_full, global_ratio_full)
    test  = attach_d49_features(test,  ratios_full, p4_ratios_full, global_ratio_full)

    # 8 — Target encoding
    train, test = add_target_encoding(train, test, cv_splits, d48_src)

    # 9 — Aggregate stats
    train, test = add_aggregate_stats(train, test, d48_src)

    # 10 — Encode categoricals
    train, test, encoders = encode_categoricals(train, test)

    # 11 — Train
    models, oof_preds, test_preds, feat_cols, oof_all, oof_d49 = train_and_predict(
        train, test, cv_splits, d48_src, encoders,
        drop_suspect_features=DROP_SUSPECT_FEATURES_FOR_TRAINING,
    )

    # 12 — Save models
    print("\n[7/8] Saving models...")
    model_path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "models":       models,
            "feature_cols": feat_cols,
            "encoders":     encoders,
            "oof_all":      oof_all,
            "oof_d49":      oof_d49,
            "oof_score":    oof_d49 if oof_d49 is not None else oof_all,
        }, f)
    print(f"  -> {model_path}")

    # 13 — Save submission
    print("\n[8/8] Saving submission...")
    submission = pd.DataFrame({"Index": test["Index"], "demand": test_preds})
    sub_path   = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  -> {sub_path}  shape={submission.shape}")

    elapsed = time.time() - t_start
    print(f"\n  Total time       : {elapsed/60:.1f} min")
    print(f"  OOF (all data)   : {oof_all:.4f}")
    if oof_d49:
        print(f"  OOF (D49 only)   : {oof_d49:.4f}  <- best proxy for online score")
    print("=" * 65)


if __name__ == "__main__":
    main()
