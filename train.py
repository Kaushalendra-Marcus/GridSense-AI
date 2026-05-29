"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction (v13 - Scientific Drift)
=============================================================================
Root-cause fixes over v12.1:
 1. Drift clipping: [0.7,1.5] → [0.3,5.0]   (54% of geohashes had drift > 1.5 — all clamped wrongly)
 2. Drift computed from slots 5-8 ONLY (1am-2am, std=8 vs std=164 for midnight slots)
 3. Timeslot-aware drift decay: drift decreases from 2.5x at midnight → ~1.0x daytime
    Applied as: effective_drift = geohash_ratio * ts_decay_factor[time_slot]
 4. Sample weight 5x for Day-49 rows (effective D49:D48 = 0.57:1 vs 0.23:1 before)
 5. Remove raw `day` feature (it's always 48 in D48 train and 49 in test — distribution shift)
 6. Target encoding uses ONLY Day-49 rows as source for test mapping
    (test is Day 49, so encoding from D49 is more aligned)
"""

import os, warnings, pickle, time, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
N_FOLDS    = 5
SEED       = 42

LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     5000,
    "learning_rate":    0.02,
    "num_leaves":       63,
    "max_depth":        7,
    "min_child_samples": 20,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.2,
    "reg_lambda":       5.0,
    "min_split_gain":   0.01,
    "random_state":     SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ─────────────────────────────────────────────
def load_data():
    logger.info("[1/8] Loading data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test      = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train     = train_raw.copy().reset_index(drop=True)
    logger.info(f"  Raw train: {train_raw.shape}  |  Test: {test.shape}")
    logger.info(f"  Day 48: {(train_raw['day']==48).sum()} rows | Day 49: {(train_raw['day']==49).sum()} rows")
    return train_raw, train, test

# ─────────────────────────────────────────────
def add_temporal_features(df):
    ts = df["timestamp"].astype(str)
    df["hour"]      = ts.str.split(":").str[0].astype(int)
    df["minute"]    = ts.str.split(":").str[1].astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    # is_day49 instead of raw 'day' — avoids distribution shift (test always = 49)
    df["is_day49"]  = (df["day"] == 49).astype(int)
    df["is_peak"]   = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]  = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_morning"]= df["hour"].isin([6, 7, 8, 9, 10]).astype(int)
    df["sin_hour"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_ts"]    = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["cos_ts"]    = np.cos(2 * np.pi * df["time_slot"] / 96)
    return df

# ─────────────────────────────────────────────
def add_geo_features(df):
    gh = df["geohash"].astype(str)
    df["geo_p3"] = gh.str[:3]
    df["geo_p4"] = gh.str[:4]
    df["geo_p5"] = gh.str[:5]
    df["geo_p6"] = gh.str[:6]
    try:
        import pygeohash as pgh
        coords = gh.apply(lambda x: pgh.decode(x) if len(x) >= 4 else (np.nan, np.nan))
        df["lat"] = coords.apply(lambda x: x[0])
        df["lon"] = coords.apply(lambda x: x[1])
    except ImportError:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df

# ─────────────────────────────────────────────
def fill_missing(train, test, src):
    for col in ["RoadType", "Weather"]:
        for df in [train, test]:
            geo_mode = (src.dropna(subset=[col])
                          .groupby("geo_p4")[col]
                          .agg(lambda x: x.mode()[0] if len(x) else np.nan))
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "geo_p4"].map(geo_mode)
            df[col] = df[col].fillna(src[col].mode()[0])
    for df in [train, test]:
        geo_day_med = src.dropna(subset=["Temperature"]).groupby(["geohash","day"])["Temperature"].median()
        df["Temperature"] = df["Temperature"].fillna(
            df.apply(lambda r: geo_day_med.get((r["geohash"], r["day"]), np.nan), axis=1))
        df["Temperature"] = df["Temperature"].fillna(src["Temperature"].median())
    return train, test

# ─────────────────────────────────────────────
def add_lag_and_drift(train_raw, train, test):
    logger.info("[3/8] Building Lag & Scientific Drift Features...")

    d48 = train_raw[train_raw["day"] == 48].copy()
    d49 = train_raw[train_raw["day"] == 49].copy()

    # ── Primary lag: demand_d48 at exact geohash+timestamp ───────────────
    d48_map = d48.set_index(["geohash", "timestamp"])["demand"]

    for df in [train, test]:
        df["demand_d48"] = df.apply(
            lambda r: d48_map.get((r["geohash"], r["timestamp"]), np.nan), axis=1)

    # Fill NaN demand_d48 with geohash median → geo_p4+ts median → global median
    geo_med   = d48.groupby("geohash")["demand"].median()
    geo_p4_ts = d48.groupby(["geo_p4","timestamp"])["demand"].median()
    ts_med    = d48.groupby("timestamp")["demand"].median()
    global_med= d48["demand"].median()

    for df in [train, test]:
        df["demand_d48"] = df["demand_d48"].fillna(df["geohash"].map(geo_med))
        mask = df["demand_d48"].isna()
        if mask.any():
            df.loc[mask, "demand_d48"] = df.loc[mask].apply(
                lambda r: geo_p4_ts.get((r["geohash"][:4], r["timestamp"]), np.nan), axis=1)
        df["demand_d48"] = df["demand_d48"].fillna(df["timestamp"].map(ts_med))
        df["demand_d48"] = df["demand_d48"].fillna(global_med)

    # ── Neighbour lags on Day 48 ─────────────────────────────────────────
    d48_ts = d48.sort_values(["geohash","time_slot"]).copy()
    d48_ts["d48_prev"] = d48_ts.groupby("geohash")["demand"].shift(1)
    d48_ts["d48_next"] = d48_ts.groupby("geohash")["demand"].shift(-1)
    d48_ts["d48_roll3"]= d48_ts.groupby("geohash")["demand"].transform(
        lambda x: x.rolling(3, center=True, min_periods=1).mean())
    d48_ts["d48_roll5"]= d48_ts.groupby("geohash")["demand"].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).mean())

    for col in ["d48_prev", "d48_next", "d48_roll3", "d48_roll5"]:
        m = d48_ts.set_index(["geohash","timestamp"])[col]
        for df in [train, test]:
            df[col] = df.apply(lambda r: m.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
            df[col] = df[col].fillna(df["demand_d48"])

    # ── SCIENTIFIC DRIFT (fix #1 + #2 + #3) ─────────────────────────────
    #
    # Fix #2: Use only slots 5-8 (1am-2am) for the per-geohash ratio.
    #         These have std=8 vs std=164 for midnight slots.
    d49_late = d49[d49["time_slot"].between(5, 8)]
    d48_late = d48[d48["time_slot"].between(5, 8)]
    d49_geo  = d49_late.groupby("geohash")["demand"].median()
    d48_geo  = d48_late.groupby("geohash")["demand"].median()
    common   = d49_geo.index.intersection(d48_geo.index)
    geo_drift = (d49_geo[common] / (d48_geo[common] + 1e-6))

    # Fix #1: Clip [0.3, 5.0] — 54% of geos have drift > 1.5, old clip was wrong
    geo_drift = geo_drift.clip(0.3, 5.0)

    # geo_p4 fallback
    d49["geo_p4"] = d49["geohash"].str[:4]
    d48["geo_p4"] = d48["geohash"].str[:4]
    d49_late2 = d49[d49["time_slot"].between(5,8)]
    d48_late2 = d48[d48["time_slot"].between(5,8)]
    p4_drift = (d49_late2.groupby("geo_p4")["demand"].median()
                / (d48_late2.groupby("geo_p4")["demand"].median() + 1e-6)).clip(0.3, 5.0)

    global_drift = float(geo_drift.median())

    # Fix #3: Timeslot decay curve
    # Observed median ratios per slot (from data analysis):
    #   slot 0→2.56, 4→1.75, 8→1.28
    # Fit exponential decay: ratio(ts) = 1.0 + A*exp(-k*ts)
    # With slot8=1.28: A*exp(-8k)=0.28; slot0=2.56: A=1.56 → exp(-8k)=0.179 → k=0.215
    # For test slots 9-55, ratio continues decaying toward 1.0
    def slot_decay(ts, k=0.215, A=1.56, baseline=1.0):
        return baseline + A * np.exp(-k * ts)

    # Normalize so that slot-5 to slot-8 mean = 1.0 (geohash drift is already calibrated there)
    norm_base = np.mean([slot_decay(s) for s in range(5, 9)])  # ~1.28
    decay_factors = {ts: slot_decay(ts) / norm_base for ts in range(96)}
    # decay_factor > 1 for midnight (ts<5), ~1 for calibration window (ts=5-8), <1 would be < 1 for daytime
    # But clamp decay to [0.5, 2.0] to avoid extreme corrections
    decay_factors = {k: max(0.5, min(2.0, v)) for k, v in decay_factors.items()}

    for df in [train, test]:
        # Per-geohash drift
        dr = df["geohash"].map(geo_drift)
        mask = dr.isna()
        dr[mask] = df.loc[mask, "geo_p4"].map(p4_drift)
        dr = dr.fillna(global_drift)
        df["drift_ratio"] = dr

        # Timeslot-adjusted drift
        ts_decay = df["time_slot"].map(decay_factors).fillna(1.0)
        df["drift_ratio_ts"] = (dr * ts_decay).clip(0.3, 6.0)

        # Primary corrected prediction
        df["demand_d48_corrected"]    = df["demand_d48"]   * df["drift_ratio"]
        df["demand_d48_corrected_ts"] = df["demand_d48"]   * df["drift_ratio_ts"]
        df["demand_d48_roll3_corr"]   = df["d48_roll3"]    * df["drift_ratio"]

    # ── Day-49 direct demand features (leak-free: slots 0-8, test starts at 9) ─
    d49_map = d49.set_index(["geohash","timestamp"])["demand"]
    for df in [train, test]:
        df["demand_d49_obs"] = df.apply(
            lambda r: d49_map.get((r["geohash"], r["timestamp"]), np.nan), axis=1)

    d49_geo_mean = d49.groupby("geohash")["demand"].mean()
    d49_geo_last = d49[d49["time_slot"]==8].groupby("geohash")["demand"].mean()
    for df in [train, test]:
        df["demand_d49_mean"] = df["geohash"].map(d49_geo_mean).fillna(
            df["geo_p4"].map(d49.groupby("geo_p4")["demand"].mean()))
        df["demand_d49_last"] = df["geohash"].map(d49_geo_last).fillna(
            df["demand_d49_mean"])
        df["demand_d49_obs"]  = df["demand_d49_obs"].fillna(df["demand_d49_mean"])

    # ===== NEW SLOPE FEATURES =====
    for df in [train, test]:
        df["d48_slope_prev"] = df["demand_d48"] - df["d48_prev"]
        df["d48_slope_next"] = df["d48_next"] - df["demand_d48"]

        df["d48_velocity"] = df["d48_roll3"] - df["d48_prev"]

        df["d48_acceleration"] = (
            df["d48_next"]
            - 2 * df["demand_d48"]
            + df["d48_prev"]
        )

        df["drifted_velocity"] = (
            df["d48_velocity"]
            * df["drift_ratio_ts"]
        )

        df["drifted_acceleration"] = (
            df["d48_acceleration"]
            * df["drift_ratio_ts"]
        )

    logger.info(f"  Global Drift (Median): {global_drift:.3f}")
    logger.info(f"  Drift Range (before clip): [{d49_geo[common].div(d48_geo[common]+1e-6).min():.2f}, "
                f"{d49_geo[common].div(d48_geo[common]+1e-6).max():.2f}]")
    logger.info(f"  NaN demand_d48 in test: {test['demand_d48'].isna().sum()}")
    return train, test

# ─────────────────────────────────────────────
def add_target_encoding(train, test, kf, src_d49, src_all):
    """
    Dual-source target encoding:
    - For fold OOF: use in-fold training rows only (standard)
    - For test: use Day-49 rows as source (aligned with test distribution)
    """
    logger.info("[4/8] Target encoding (OOF)...")
    global_mean_all = float(train["demand"].mean())
    global_mean_d49 = float(src_d49["demand"].mean())

    encode_keys = [
        "geohash",
        "geo_p4",
        "geo_p5",
        "geo_p6",

        ("geohash", "time_slot"),
        ("geohash", "hour"),

        ("geo_p4", "time_slot"),
        ("geo_p5", "time_slot"),
        ("geo_p6", "time_slot"),

        ("RoadType", "time_slot"),
        ("Weather", "time_slot"),

        ("RoadType", "hour"),
        ("Weather", "hour"),

        ("geohash", "RoadType"),
        ("geohash", "Weather"),

        ("geo_p6", "RoadType"),
        ("geo_p6", "Weather"),

        ("geohash", "is_peak"),
    ]

    def make_key(df, cols):
        return df[cols].fillna(-1).astype(str).agg("_".join, axis=1)

    for key in encode_keys:
        cols = [key] if isinstance(key, str) else list(key)
        col_name = ("_x_".join(cols)) + "_enc"

        train["_key"] = make_key(train, cols)
        test["_key"]  = make_key(test,  cols)
        src_d49["_key"] = make_key(src_d49, cols)
        src_all["_key"] = make_key(src_all, cols)

        # OOF for train
        train[col_name] = np.nan
        for _, (tr_idx, val_idx) in enumerate(kf.split(train)):
            mapping = train.iloc[tr_idx].groupby("_key")["demand"].mean()
            train.loc[val_idx, col_name] = train.loc[val_idx, "_key"].map(mapping)
        train[col_name] = train[col_name].fillna(global_mean_all)

        # Test: prefer D49-derived encoding (same day as test)
        map_d49 = src_d49.groupby("_key")["demand"].mean()
        map_all = src_all.groupby("_key")["demand"].mean()
        test[col_name] = test["_key"].map(map_d49)
        test[col_name] = test[col_name].fillna(test["_key"].map(map_all))
        test[col_name] = test[col_name].fillna(global_mean_d49)

    for df in [train, test, src_d49, src_all]:
        if "_key" in df.columns:
            df.drop(columns=["_key"], inplace=True)

    return train, test

# ─────────────────────────────────────────────
def add_aggregate_stats(train, test, src_d49, src_all):
    logger.info("[5/8] Aggregate statistics...")

    agg_configs = [
        ("geohash",       ["mean","std","median","max","min"]),
        ("time_slot",     ["mean","std"]),
        ("geo_p4",        ["mean","std","median"]),
        ("RoadType",      ["mean"]),
        ("NumberofLanes", ["mean","std"]),
        ("Weather",       ["mean"]),
        ("hour",          ["mean","std"]),
    ]

    # Use ALL training data for agg stats (more stable than D49 only)
    for key, aggs in agg_configs:
        stats = (src_all.groupby(key)["demand"]
                 .agg(aggs).add_prefix(f"{key}_").reset_index())
        train = train.merge(stats, on=key, how="left")
        test  = test.merge(stats,  on=key, how="left")

    eps = 1e-6
    for df in [train, test]:
        df["lag_to_geo_mean"]       = df["demand_d48"]           / (df["geohash_mean"] + eps)
        df["corrected_to_geo"]      = df["demand_d48_corrected"] / (df["geohash_mean"] + eps)
        df["corrected_ts_to_geo"]   = df["demand_d48_corrected_ts"] / (df["geohash_mean"] + eps)
        df["lag_minus_geo_median"]  = df["demand_d48"] - df["geohash_median"]
        df["d49mean_to_geo"]        = df["demand_d49_mean"] / (df["geohash_mean"] + eps)
        df["temp_bin"] = pd.cut(df["Temperature"],
                                bins=[-999,15,22,30,999], labels=[0,1,2,3]).astype(float)

    train_meds = train.select_dtypes(include=[np.number]).median()
    train = train.fillna(train_meds)
    test  = test.fillna(train_meds)
    return train, test

# ─────────────────────────────────────────────
def prepare_categoricals(train, test):
    cat_cols = ["geohash","geo_p3","geo_p4","geo_p5","geo_p6",
                "RoadType","Weather","LargeVehicles","Landmarks"]
    cat_cols = [c for c in cat_cols if c in train.columns]
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        cat_type = pd.CategoricalDtype(categories=combined.unique(), ordered=False)
        train[col] = train[col].astype(str).astype(cat_type)
        test[col]  = test[col].astype(str).astype(cat_type)
    return train, test, cat_cols

# ─────────────────────────────────────────────
def train_and_predict(train, test, kf, cat_cols):
    logger.info("[6/8] Training LightGBM (5-fold CV)...")

    # Drop leaky/redundant cols
    drop_cols    = ["Index","demand","timestamp","hour","minute","day"]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    active_cats  = [c for c in cat_cols if c in feature_cols]
    LGBM_PARAMS["categorical_feature"] = active_cats

    X      = train[feature_cols]
    y      = train["demand"]
    X_test = test[feature_cols]

    # Fix #4: weight 5x for Day-49 rows (effective ratio D49:D48 = 0.57:1)
    sample_weights = np.where(train["day"] == 49, 5.0, 1.0)

    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Cat features: {active_cats}")

    oof_preds  = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    models     = []
    fold_scores= []
    d49_mask   = (train["day"] == 49).values

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
        oof_preds[val_idx]  = model.predict(X.iloc[val_idx])
        test_preds         += model.predict(X_test) / N_FOLDS

        fold_r2    = r2_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_score = max(0, 100 * fold_r2)
        fold_scores.append(fold_score)
        models.append(model)

        d49_val = d49_mask[val_idx]
        if d49_val.sum() > 0:
            d49_score = max(0, 100 * r2_score(y.iloc[val_idx][d49_val],
                                               oof_preds[val_idx][d49_val]))
            logger.info(f"  Fold {fold+1} | OOF: {fold_score:.2f} | D49-OOF: {d49_score:.2f} | "
                        f"Trees: {model.best_iteration_} | Time: {time.time()-t0:.1f}s")
        else:
            logger.info(f"  Fold {fold+1} | OOF: {fold_score:.2f} | "
                        f"Trees: {model.best_iteration_} | Time: {time.time()-t0:.1f}s")

    oof_score = max(0, 100 * r2_score(y, oof_preds))
    logger.info(f"\n  ─ OOF Score (all data):  {oof_score:.4f} ─")
    if d49_mask.sum() > 0:
        d49_oof = max(0, 100 * r2_score(y[d49_mask], oof_preds[d49_mask]))
        logger.info(f"  ── OOF Score (Day 49):    {d49_oof:.4f}  ← best proxy for online score ──")
    logger.info(f"  ── Fold Scores: {[f'{s:.2f}' for s in fold_scores]} ──")

    # Feature importance
    importances = np.zeros(len(feature_cols))
    for m in models:
        importances += m.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    logger.info(f"  Top features: {imp_df['feature'].head(10).tolist()}")

    return models, oof_preds, test_preds, feature_cols, oof_score

# ─────────────────────────────────────────────
def train_ratio_model(train, test, feature_cols):
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    ratio_target = train["demand"] / (train["demand_d48"] + 1e-6)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X, ratio_target)

    ratio_pred = model.predict(X_test)
    ratio_pred = np.clip(ratio_pred, 0.2, 8.0)

    return test["demand_d48"].values * ratio_pred

# ─────────────────────────────────────────────
def train_catboost_model(train, test, feature_cols, active_cats):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.info("  CatBoost not installed; skipping CatBoost ensemble.")
        return None

    catboost_features = [c for c in active_cats if c in feature_cols]
    catboost_cat_idx = [feature_cols.index(c) for c in catboost_features]

    model = CatBoostRegressor(
        iterations=3500,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=200,
    )

    model.fit(
        train[feature_cols],
        train["demand"],
        cat_features=catboost_cat_idx if catboost_cat_idx else None,
    )

    return model.predict(test[feature_cols])

# ─────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction v13")
    logger.info("=" * 60)

    train_raw, train, test = load_data()

    logger.info("\n[2/8] Feature engineering...")
    for df in [train_raw, train, test]:
        add_temporal_features(df)
        add_geo_features(df)

    src_all = train_raw.copy()  # all days, for aggregate stats and encoding fallback
    src_d49 = train_raw[train_raw["day"] == 49].copy()  # D49 only, for test-aligned encoding

    train, test = fill_missing(train, test, src=src_all)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    train, test = add_lag_and_drift(train_raw, train, test)
    train, test = add_target_encoding(train, test, kf, src_d49=src_d49, src_all=src_all)
    train, test = add_aggregate_stats(train, test, src_d49=src_d49, src_all=src_all)
    train, test, cat_cols = prepare_categoricals(train, test)

    models, oof_preds, test_preds, feature_cols, oof_score = train_and_predict(
        train, test, kf, cat_cols)

    ratio_preds = train_ratio_model(train, test, feature_cols)
    cb_preds = train_catboost_model(train, test, feature_cols, cat_cols)

    if cb_preds is not None:
        final_preds = (0.45 * test_preds + 0.30 * cb_preds + 0.25 * ratio_preds)
    else:
        final_preds = (0.60 * test_preds + 0.40 * ratio_preds)

    save_path = os.path.join(MODEL_DIR, "lgbm_models.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols,
                     "cat_cols": cat_cols, "oof_score": oof_score}, f)
    logger.info(f"\n[7/8] Models saved → {save_path}")

    submission = pd.DataFrame({"Index": test["Index"], "demand": final_preds})
    sub_path   = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    logger.info(f"       Submission → {sub_path}  shape: {submission.shape}")

    logger.info(f"\n  Total time:      {(time.time()-t_start)/60:.1f} min")
    logger.info(f"  Final OOF Score: {oof_score:.4f} / 100")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()