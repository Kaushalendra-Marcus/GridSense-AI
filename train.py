"""
Gridlock Hackathon 2.0 — Traffic Demand Prediction
====================================================
v6 — Targeted improvements over v5:

  SCORE HISTORY:
    v2: 89.63  (fixed aggregate leakage + params)
    v3: 90.63  (D49 drift ratio, D48+D49 training, d49-oof reporting)
    v5: 90.85  (lat/lon from geohash, bug fixes)

  v6 CHANGES:
  1. Lag fallback improved: use geohash x time_slot median from D48
     BEFORE geohash-only median. For the 11.1% of test rows that miss
     the exact lag, this preserves the time-of-day signal.
  2. Bayesian smoothing on d49_d48_ratio:
     shrink noisy 1-2 sample geohashes toward the global ratio.
  3. Adjacent-slot D48 features: demand at slot-1, slot+1, slot-2,
     slot+2 for each geohash. Gives the model a demand-trend slope.
  4. D49_SAMPLE_WEIGHT 4 -> 3  (4x caused over-emphasis on 9 early
     D49 slots which are not representative of test time window).
  5. Increased colsample_bytree to 0.8 (more features sampled per tree
     since feature set is now richer with 60+ features).

Run:  python train.py
Out:  outputs/submission.csv  |  models/lgbm_models.pkl
"""

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
MODEL_DIR         = "models"
OUTPUT_DIR        = "outputs"
SEED              = 42
N_FOLDS           = 5
D49_SAMPLE_WEIGHT = 3.0   # v6: reduced from 4 (D49-train only covers early slots)

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
    "colsample_bytree":  0.8,    # v6: up from 0.75 (more features now)
    "reg_alpha":         0.05,
    "reg_lambda":        0.15,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# GEOHASH DECODER  (pure Python, no external dependency)
# ─────────────────────────────────────────────────────────────────────
_B32MAP = {c: i for i, c in enumerate("0123456789bcdefghjkmnpqrstuvwxyz")}

def _decode_geohash(gh):
    lat_lo, lat_hi = -90.0,  90.0
    lon_lo, lon_hi = -180.0, 180.0
    is_lon = True
    for char in str(gh).lower():
        if char not in _B32MAP:
            continue
        bits = _B32MAP[char]
        for i in range(4, -1, -1):
            bit = (bits >> i) & 1
            if is_lon:
                mid = (lon_lo + lon_hi) / 2
                if bit: lon_lo = mid
                else:   lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if bit: lat_lo = mid
                else:   lat_hi = mid
            is_lon = not is_lon
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

def build_latlon_lookup(geohashes):
    result = {}
    for gh in geohashes:
        try:
            result[gh] = _decode_geohash(gh)
        except Exception:
            result[gh] = (np.nan, np.nan)
    return result


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1/9] Loading data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test      = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    d48 = train_raw[train_raw["day"] == 48].copy().reset_index(drop=True)
    d49 = train_raw[train_raw["day"] == 49].copy().reset_index(drop=True)
    print(f"  D48: {len(d48)} rows | D49: {len(d49)} rows | Test: {len(test)} rows")
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
    df["is_peak"]   = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]  = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_day49"]  = (df["day"] == 49).astype(int)
    df["sin_slot"]  = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["cos_slot"]  = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["sin_hour"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — GEO FEATURES  (prefix hierarchy + lat/lon)
# ─────────────────────────────────────────────────────────────────────

def add_geo_features(df, latlon_lookup):
    gh = df["geohash"].astype(str)
    df = df.copy()
    df["geo_p3"] = gh.str[:3]
    df["geo_p4"] = gh.str[:4]
    df["geo_p5"] = gh.str[:5]
    df["geo_p6"] = gh.str[:6]
    df["lat"]    = [latlon_lookup.get(g, (np.nan, np.nan))[0] for g in df["geohash"]]
    df["lon"]    = [latlon_lookup.get(g, (np.nan, np.nan))[1] for g in df["geohash"]]
    df["lat_x_slot"] = df["lat"] * df["time_slot"]
    df["lon_x_slot"] = df["lon"] * df["time_slot"]
    df["lat_x_lon"]  = df["lat"] * df["lon"]
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — MISSING VALUE FILL  (geohash-structural from EDA)
# ─────────────────────────────────────────────────────────────────────

def fill_missing(train, test, d48_src):
    train = train.copy()
    test  = test.copy()

    for col in ["RoadType", "Weather"]:
        gh_mode = (d48_src.dropna(subset=[col])
                   .groupby("geohash")[col]
                   .agg(lambda x: x.mode().iloc[0]))
        p4_mode = (d48_src.dropna(subset=[col])
                   .groupby("geo_p4")[col]
                   .agg(lambda x: x.mode().iloc[0]))
        global_mode = d48_src[col].mode().iloc[0]
        for df in [train, test]:
            m = df[col].isna()
            df.loc[m, col] = df.loc[m, "geohash"].map(gh_mode)
            m = df[col].isna()
            df.loc[m, col] = df.loc[m, "geo_p4"].map(p4_mode)
            df[col] = df[col].fillna(global_mode)

    gh_temp = d48_src.dropna(subset=["Temperature"]).groupby("geohash")["Temperature"].median()
    g_temp  = d48_src["Temperature"].median()
    for df in [train, test]:
        m = df["Temperature"].isna()
        df.loc[m, "Temperature"] = df.loc[m, "geohash"].map(gh_temp)
        df["Temperature"] = df["Temperature"].fillna(g_temp)

    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — LAG FEATURES
#
# v6 CHANGE — Better fallback chain:
#   1. Exact geohash + timestamp match     (is_lag_real = 1)
#   2. Geohash + time_slot median (D48)    <- NEW: keeps time-of-day signal
#   3. geo_p4 + timestamp median
#   4. timestamp median
#   5. global median
# ─────────────────────────────────────────────────────────────────────

def add_lag_features(d48, d49_train, test):
    print("\n[3/9] Building lag features...")

    # D48 lookup tables
    d48_exact      = d48.set_index(["geohash", "timestamp"])["demand"]
    gh_slot_med    = d48.groupby(["geohash", "time_slot"])["demand"].median()   # v6: primary fallback
    gh_med         = d48.groupby("geohash")["demand"].median()
    p4_ts_med      = d48.groupby(["geo_p4", "timestamp"])["demand"].median()
    ts_med         = d48.groupby("timestamp")["demand"].median()
    global_med     = float(d48["demand"].median())

    # ── D48 self-lag (proxy: same geohash × slot in D48) ─────────────
    d48 = d48.copy()
    d48["demand_d48"]  = list(d48.apply(
        lambda r: gh_slot_med.get((r["geohash"], r["time_slot"]), np.nan), axis=1))
    m = d48["demand_d48"].isna()
    d48.loc[m, "demand_d48"] = d48.loc[m, "time_slot"].map(
        d48.groupby("time_slot")["demand"].median())
    d48["demand_d48"]  = d48["demand_d48"].fillna(global_med)
    d48["is_lag_real"] = 0

    # ── Real lag for D49 / test (improved fallback chain) ─────────────
    def attach_lag(df):
        df = df.copy()
        # 1. Exact match
        df["demand_d48"] = [d48_exact.get((g, t), np.nan)
                            for g, t in zip(df["geohash"], df["timestamp"])]
        df["is_lag_real"] = df["demand_d48"].notna().astype(int)

        # 2. v6: geohash × time_slot median (keeps time-of-day signal)
        m = df["demand_d48"].isna()
        if m.any():
            df.loc[m, "demand_d48"] = df.loc[m].apply(
                lambda r: gh_slot_med.get((r["geohash"], r["time_slot"]), np.nan), axis=1)

        # 3. geohash median
        m = df["demand_d48"].isna()
        df.loc[m, "demand_d48"] = df.loc[m, "geohash"].map(gh_med)

        # 4. geo_p4 + timestamp median
        m = df["demand_d48"].isna()
        if m.any():
            df.loc[m, "demand_d48"] = df.loc[m].apply(
                lambda r: p4_ts_med.get((r["geo_p4"], r["timestamp"]), np.nan), axis=1)

        # 5. timestamp median
        m = df["demand_d48"].isna()
        df.loc[m, "demand_d48"] = df.loc[m, "timestamp"].map(ts_med)

        # 6. global median
        df["demand_d48"] = df["demand_d48"].fillna(global_med)
        return df

    d49_train = attach_lag(d49_train)
    test      = attach_lag(test)

    print(f"  D49-train exact lag coverage: {d49_train['is_lag_real'].mean()*100:.1f}%")
    print(f"  Test      exact lag coverage: {test['is_lag_real'].mean()*100:.1f}%")
    return d48, d49_train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 5b — ADJACENT SLOT D48 FEATURES  (v6 new)
#
# For each row, look up D48 demand at the same geohash but
# slot-2, slot-1, slot+1, slot+2.  This gives the model
# a local demand-trend slope (rising vs falling demand).
# ─────────────────────────────────────────────────────────────────────

def add_adjacent_slot_features(train, test, d48_src):
    print("\n[3c/9] Adjacent-slot D48 features...")

    gh_slot_med = d48_src.groupby(["geohash", "time_slot"])["demand"].median()
    p4_slot_med = d48_src.groupby(["geo_p4",  "time_slot"])["demand"].median()
    slot_med    = d48_src.groupby("time_slot")["demand"].median()

    def get_adj(df, offsets=(-2, -1, 1, 2)):
        df = df.copy()
        for off in offsets:
            col = f"d48_slot_{'+' if off>0 else ''}{off}"
            vals = []
            for _, row in df.iterrows():
                s = int(row["time_slot"]) + off
                if s < 0 or s > 95:
                    vals.append(np.nan)
                    continue
                v = gh_slot_med.get((row["geohash"], s), np.nan)
                if np.isnan(v):
                    v = p4_slot_med.get((row["geo_p4"], s), np.nan)
                if np.isnan(v):
                    v = slot_med.get(s, np.nan)
                vals.append(v)
            df[col] = vals

        # Demand trend slope: (slot+1 - slot-1) / 2
        df["d48_demand_slope"] = (
            df.get("d48_slot_+1", np.nan) - df.get("d48_slot_-1", np.nan)
        ) / 2.0
        return df

    train = get_adj(train)
    test  = get_adj(test)

    # Fill NaN adjacent slots with the row's own demand_d48
    adj_cols = [c for c in train.columns if c.startswith("d48_slot_")]
    for df in [train, test]:
        for col in adj_cols + ["d48_demand_slope"]:
            df[col] = df[col].fillna(df["demand_d48"])

    print(f"  Added {len(adj_cols)+1} adjacent-slot features")
    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 6 — D49 DRIFT RATIO
#
# v6 CHANGE — Bayesian smoothing:
#   Shrink per-geohash ratio toward global when n_samples is small.
#   smoothed = (n * observed + alpha * global) / (n + alpha)
#   alpha=8 means we need ~8 samples to trust the per-geohash ratio.
# ─────────────────────────────────────────────────────────────────────

def compute_d49_ratio(d49_rows, d48_src, clip=(0.5, 3.0), smooth_alpha=8):
    """
    Per-geohash D49/D48 drift ratio with Bayesian smoothing.
    Later slots get higher weight (closer to test window).
    ALWAYS returns (geohash_series, geo_p4_series, global_float).
    """
    EMPTY, DEFAULT = pd.Series(dtype=float), 1.0

    d48_map = d48_src.set_index(["geohash", "timestamp"])["demand"]
    d49r    = d49_rows.copy()
    d49r["demand_d48_real"] = [d48_map.get((g, t), np.nan)
                               for g, t in zip(d49r["geohash"], d49r["timestamp"])]
    matched = d49r.dropna(subset=["demand_d48_real"]).copy()
    if len(matched) == 0:
        return EMPTY, EMPTY, DEFAULT

    matched["slot_weight"] = matched["time_slot"] + 1   # slots 0-8 -> weights 1-9

    def geo_ratio(grp):
        w = grp["slot_weight"].values.astype(float)
        d49w = np.average(grp["demand"].values,          weights=w)
        d48w = np.average(grp["demand_d48_real"].values, weights=w)
        return d49w / d48w if d48w > 1e-6 else np.nan

    raw_ratios  = matched.groupby("geohash").apply(geo_ratio).dropna()
    n_per_geo   = matched.groupby("geohash").size()
    global_r    = float(raw_ratios.median()) if len(raw_ratios) > 0 else DEFAULT

    # Bayesian smoothing: shrink toward global_r for sparse geohashes
    smoothed = {}
    for gh, obs_r in raw_ratios.items():
        n = n_per_geo.get(gh, 1)
        smoothed[gh] = (n * obs_r + smooth_alpha * global_r) / (n + smooth_alpha)
    ratios = pd.Series(smoothed).clip(*clip)

    # geo_p4 level (smoothed similarly)
    if "geo_p4" not in matched.columns:
        matched["geo_p4"] = matched["geohash"].str[:4]
    raw_p4  = matched.groupby("geo_p4").apply(geo_ratio).dropna()
    n_p4    = matched.groupby("geo_p4").size()
    p4_smoothed = {}
    for p4, obs_r in raw_p4.items():
        n = n_p4.get(p4, 1)
        p4_smoothed[p4] = (n * obs_r + smooth_alpha * global_r) / (n + smooth_alpha)
    p4_ratios = pd.Series(p4_smoothed).clip(*clip)

    return ratios, p4_ratios, global_r


def attach_d49_features(df, ratios, p4_ratios, global_ratio):
    df = df.copy()
    gh_ratio = df["geohash"].map(ratios)
    m = gh_ratio.isna()
    if m.any():
        gh_ratio[m] = df.loc[m, "geo_p4"].map(p4_ratios)
    gh_ratio = gh_ratio.fillna(global_ratio)
    df["d49_d48_ratio"]        = gh_ratio.values
    df["demand_d48_corrected"] = df["demand_d48"] * gh_ratio.values
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 7 — AGGREGATE STATS  (D48 source only — zero leakage)
# ─────────────────────────────────────────────────────────────────────

def add_aggregate_stats(train, test, d48_src):
    print("\n[5/9] Aggregate statistics (D48 source)...")

    configs = [
        ("geohash",       ["mean", "std", "median", "max", "min"]),
        ("time_slot",     ["mean", "std"]),
        ("geo_p4",        ["mean", "std"]),
        ("geo_p5",        ["mean"]),
        ("RoadType",      ["mean"]),
        ("NumberofLanes", ["mean"]),
        ("Weather",       ["mean"]),
    ]
    for key, aggs in configs:
        stats = (d48_src.groupby(key)["demand"]
                 .agg(aggs).add_prefix(f"{key}_d48_").reset_index())
        train = train.merge(stats, on=key, how="left")
        test  = test.merge(stats,  on=key, how="left")

    # Geohash × test-slot stats from D48 (slots 9-55 = test window)
    ts_d48 = d48_src[d48_src["time_slot"].between(9, 55)]
    if len(ts_d48) > 0:
        gh_ts = (ts_d48.groupby(["geohash", "time_slot"])["demand"]
                 .agg(["mean", "std"]).reset_index()
                 .rename(columns={"mean": "gh_ts_d48_mean", "std": "gh_ts_d48_std"}))
        p4_ts = (ts_d48.groupby(["geo_p4", "time_slot"])["demand"]
                 .mean().reset_index().rename(columns={"demand": "p4_ts_d48_mean"}))
        train = train.merge(gh_ts, on=["geohash", "time_slot"], how="left")
        test  = test.merge(gh_ts,  on=["geohash", "time_slot"], how="left")
        train = train.merge(p4_ts, on=["geo_p4", "time_slot"], how="left")
        test  = test.merge(p4_ts,  on=["geo_p4", "time_slot"], how="left")

    eps = 1e-6
    for df in [train, test]:
        df["lag_to_geo_mean"]   = df["demand_d48"] / (df["geohash_d48_mean"] + eps)
        df["lag_minus_geo_med"] = df["demand_d48"] - df["geohash_d48_median"]
        df["lag_to_slot_mean"]  = df["demand_d48"] / (df["time_slot_d48_mean"] + eps)
        df["temp_bin"]          = pd.cut(df["Temperature"],
                                         bins=[-999, 15, 22, 30, 999],
                                         labels=[0, 1, 2, 3]).astype(float)

    med   = train.select_dtypes(include=[np.number]).median()
    train = train.fillna(med)
    test  = test.fillna(med)
    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 8 — TARGET ENCODING  (OOF-safe)
# ─────────────────────────────────────────────────────────────────────

def add_target_encoding(train, test, kf, d48_src):
    print("\n[4/9] Target encoding (OOF-safe)...")
    global_mean = float(train["demand"].mean())
    full_src    = pd.concat([train, d48_src], ignore_index=True)

    keys = [
        "geohash", "geo_p4", "geo_p5",
        ("geohash",  "time_slot"),
        ("geohash",  "is_day49"),
        ("geo_p4",   "time_slot"),
        ("geo_p5",   "time_slot"),
        ("day",      "time_slot"),
        ("RoadType", "time_slot"),
        ("Weather",  "time_slot"),
        ("geohash",  "is_peak"),
    ]
    for key in keys:
        if isinstance(key, tuple):
            col = "_x_".join(key) + "_enc"
            for df in [train, test, full_src]:
                df["_k"] = df[list(key)].astype(str).agg("_".join, axis=1)
        else:
            col = key + "_enc"
            for df in [train, test, full_src]:
                df["_k"] = df[key].astype(str)

        train[col] = np.nan
        for _, (tr_idx, val_idx) in enumerate(kf.split(train)):
            m = train.iloc[tr_idx].groupby("_k")["demand"].mean()
            train.loc[val_idx, col] = train.loc[val_idx, "_k"].map(m)

        full_map   = full_src.groupby("_k")["demand"].mean()
        test[col]  = test["_k"].map(full_map)
        train[col] = train[col].fillna(global_mean)
        test[col]  = test[col].fillna(global_mean)

    for df in [train, test, full_src]:
        df.drop(columns=["_k"], inplace=True, errors="ignore")

    return train, test


# ─────────────────────────────────────────────────────────────────────
# STEP 9 — LABEL ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────

def encode_categoricals(train, test):
    cats = ["geohash", "geo_p3", "geo_p4", "geo_p5", "geo_p6",
            "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    cats = [c for c in cats if c in train.columns]
    encoders = {}
    for col in cats:
        le = LabelEncoder()
        le.fit(pd.concat([train[col].astype(str), test[col].astype(str)]))
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        encoders[col] = le
    return train, test, encoders


# ─────────────────────────────────────────────────────────────────────
# STEP 10 — TRAIN + PREDICT
# ─────────────────────────────────────────────────────────────────────

def train_and_predict(train, test, kf, d48_src):
    print("\n[6/9] Training LightGBM (5-fold CV)...")

    drop  = {"Index", "demand", "timestamp", "hour", "minute"}
    fcols = [c for c in train.columns if c not in drop]
    X, y  = train[fcols], train["demand"]
    Xt    = test[fcols]

    weights = np.where(train["is_day49"].values == 1, D49_SAMPLE_WEIGHT, 1.0)
    is_d49  = train["is_day49"].values == 1

    print(f"  Features  : {len(fcols)}")
    print(f"  D48 rows  : {(~is_d49).sum()}  |  D49 rows: {is_d49.sum()}")

    # Column indices for per-fold ratio override
    ratio_col   = fcols.index("d49_d48_ratio")        if "d49_d48_ratio"        in fcols else None
    correct_col = fcols.index("demand_d48_corrected") if "demand_d48_corrected" in fcols else None
    d48_col     = fcols.index("demand_d48")           if "demand_d48"           in fcols else None

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    models, fold_scores, d49_scores = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()

        # OOF-safe ratio from training-fold D49 rows only
        tr_df    = train.iloc[tr_idx]
        d49_fold = tr_df[tr_df["is_day49"] == 1]
        r_fold, p4_fold, g_fold = compute_d49_ratio(d49_fold, d48_src)

        X_tr  = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()

        for part in [X_tr, X_val]:
            gh_ratio = part["geohash"].map(r_fold)
            m = gh_ratio.isna()
            if m.any():
                gh_ratio[m] = part.loc[m, "geo_p4"].map(p4_fold)
            gh_ratio = gh_ratio.fillna(g_fold)
            if ratio_col is not None:
                part.iloc[:, ratio_col] = gh_ratio.values
            if correct_col is not None and d48_col is not None:
                part.iloc[:, correct_col] = part.iloc[:, d48_col] * gh_ratio.values

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

        oof[val_idx]  = model.predict(X_val)
        test_preds   += model.predict(Xt) / N_FOLDS
        models.append(model)

        s = max(0, 100 * r2_score(y.iloc[val_idx], oof[val_idx]))
        fold_scores.append(s)

        val_d49 = is_d49[val_idx]
        d49_str = ""
        if val_d49.sum() > 0:
            ds = max(0, 100 * r2_score(y.iloc[val_idx][val_d49],
                                       oof[val_idx][val_d49]))
            d49_scores.append(ds)
            d49_str = f" | D49-OOF: {ds:.2f}"

        print(f"  Fold {fold+1} | OOF: {s:.2f}{d49_str} | "
              f"Trees: {model.best_iteration_} | Time: {time.time()-t0:.1f}s")

    oof_all = max(0, 100 * r2_score(y, oof))
    oof_d49 = max(0, 100 * r2_score(y[is_d49], oof[is_d49])) if is_d49.sum() > 0 else None

    print(f"\n  OOF (all data) : {oof_all:.4f}")
    if oof_d49:
        print(f"  OOF (D49 only) : {oof_d49:.4f}  <- best proxy for online score")
    print(f"  Fold scores    : {[f'{s:.2f}' for s in fold_scores]}")

    return models, oof, test_preds, fcols, oof_all, oof_d49


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction  v6")
    print("=" * 65)

    train_raw, d48, d49, test = load_data()

    print("\n[2/9] Feature engineering...")
    d48  = add_temporal_features(d48)
    d49  = add_temporal_features(d49)
    test = add_temporal_features(test)

    all_gh = set(d48["geohash"]) | set(d49["geohash"]) | set(test["geohash"])
    print(f"  Decoding lat/lon for {len(all_gh)} unique geohashes...")
    ll = build_latlon_lookup(all_gh)

    d48  = add_geo_features(d48,  ll)
    d49  = add_geo_features(d49,  ll)
    test = add_geo_features(test, ll)

    d48_src = d48.copy()
    train   = pd.concat([d48, d49], ignore_index=True)
    train, test = fill_missing(train, test, d48_src)

    d48 = train[train["day"] == 48].copy().reset_index(drop=True)
    d49 = train[train["day"] == 49].copy().reset_index(drop=True)
    d48_src = d48.copy()

    d48, d49, test = add_lag_features(d48, d49, test)
    train = pd.concat([d48, d49], ignore_index=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print("\n[3b/9] D49 drift ratio (Bayesian-smoothed)...")
    r_full, p4_full, g_full = compute_d49_ratio(d49, d48_src)
    print(f"  Global ratio: {g_full:.4f}  |  Geohashes with data: {len(r_full)}")
    train = attach_d49_features(train, r_full, p4_full, g_full)
    test  = attach_d49_features(test,  r_full, p4_full, g_full)

    train, test = add_target_encoding(train, test, kf, d48_src)

    # v6: adjacent-slot features (after lag is available on both splits)
    train, test = add_adjacent_slot_features(train, test, d48_src)

    train, test = add_aggregate_stats(train, test, d48_src)
    train, test, encoders = encode_categoricals(train, test)

    models, oof, test_preds, fcols, oof_all, oof_d49 = train_and_predict(
        train, test, kf, d48_src)

    print("\n[7/9] Saving models...")
    mp = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    with open(mp, "wb") as f:
        pickle.dump({"models": models, "feature_cols": fcols,
                     "encoders": encoders,
                     "oof_score": oof_d49 if oof_d49 else oof_all,
                     "oof_all": oof_all, "oof_d49": oof_d49}, f)
    print(f"  -> {mp}")

    print("\n[8/9] Saving submission...")
    sub = pd.DataFrame({"Index": test["Index"], "demand": test_preds})
    sp  = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(sp, index=False)
    print(f"  -> {sp}  shape={sub.shape}")

    print(f"\n[9/9] Done in {(time.time()-t0)/60:.1f} min")
    print(f"  OOF (all)  : {oof_all:.4f}")
    if oof_d49:
        print(f"  OOF (D49)  : {oof_d49:.4f}  <- best proxy for online score")
    print("=" * 65)


if __name__ == "__main__":
    main()
