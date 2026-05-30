"""
Gridlock Hackathon 2.0 — Comprehensive EDA
===========================================
Run:  python eda.py
Outputs: terminal text + CSV files in outputs/eda/

Sections
--------
 1. Data shape & schema
 2. Missing value analysis
 3. Target (demand) distribution
 4. Temporal coverage  (what timestamps exist in D48 / D49-train / test)
 5. Geohash coverage & overlap  (D48 vs D49-train vs test)
 6. Lag quality  (how well D48 demand predicts D49 demand)
 7. Day-48 to Day-49 drift  (per geohash and per time-slot)
 8. Feature analysis  (RoadType / Weather / Temp / Lanes / etc.)
 9. Demand distributions  (by hour, time-slot, RoadType, Weather)
10. Test-set characteristics  (geohash coverage, cold-start count)
11. Key findings summary  (actionable model hints)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "data"
OUT_DIR  = os.path.join("outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

DIVIDER = "=" * 70
SECTION = "-" * 70


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def parse_time(df):
    ts = df["timestamp"].astype(str)
    df = df.copy()
    df["hour"]      = ts.str.split(":").str[0].astype(int)
    df["minute"]    = ts.str.split(":").str[1].astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    return df

def pct(n, total):
    return f"{n} ({100*n/total:.1f}%)"

def fmt_stats(series):
    s = series.dropna()
    if len(s) == 0:
        return "  (no data)"
    q = s.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
    return (f"  mean={s.mean():.5f}  std={s.std():.5f}  "
            f"min={s.min():.5f}  max={s.max():.5f}\n"
            f"  p1={q[0.01]:.5f}  p25={q[0.25]:.5f}  p50={q[0.50]:.5f}  "
            f"p75={q[0.75]:.5f}  p99={q[0.99]:.5f}")

def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  -> saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────

def load():
    train_raw  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test       = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    train_raw = parse_time(train_raw)
    test      = parse_time(test)

    d48 = train_raw[train_raw["day"] == 48].copy()
    d49 = train_raw[train_raw["day"] == 49].copy()

    return train_raw, d48, d49, test, sample_sub


# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — SHAPE & SCHEMA
# ─────────────────────────────────────────────────────────────────────

def section1_shape(train_raw, d48, d49, test, sample_sub):
    print(f"\n{DIVIDER}")
    print("  SECTION 1 — Data Shape & Schema")
    print(DIVIDER)

    for name, df in [("train (all)", train_raw), ("Day 48", d48),
                     ("Day 49 (train)", d49), ("test", test),
                     ("sample_submission", sample_sub)]:
        print(f"\n  [{name}]  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")

    print(f"\n  Unique days in train : {sorted(train_raw['day'].unique())}")
    print(f"  Day 48 rows  : {len(d48)}")
    print(f"  Day 49 rows  : {len(d49)}")
    print(f"  Test rows    : {len(test)}")

    print(f"\n  Day 48  unique geohashes : {d48['geohash'].nunique()}")
    print(f"  Day 49  unique geohashes : {d49['geohash'].nunique()}")
    print(f"  Test    unique geohashes : {test['geohash'].nunique()}")

    print(f"\n  Day 48  unique timestamps: {d48['timestamp'].nunique()}")
    print(f"  Day 49  unique timestamps: {d49['timestamp'].nunique()}")
    print(f"  Test    unique timestamps: {test['timestamp'].nunique()}")

    print(f"\n  DTYPES (train):")
    for col, dt in train_raw.dtypes.items():
        print(f"    {col:<20} {str(dt)}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — MISSING VALUES
# ─────────────────────────────────────────────────────────────────────

def section2_missing(train_raw, d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 2 — Missing Value Analysis")
    print(DIVIDER)

    rows = []
    for name, df in [("D48_train", d48), ("D49_train", d49), ("test", test)]:
        n = len(df)
        for col in df.columns:
            miss = df[col].isna().sum()
            rows.append({"dataset": name, "column": col,
                         "missing": miss, "pct_missing": round(100*miss/n, 2)})

    miss_df = pd.DataFrame(rows)
    pivot = miss_df.pivot_table(
        index="column", columns="dataset",
        values="pct_missing", aggfunc="first"
    ).fillna(0)
    print(f"\n  Missing % per column:\n{pivot.round(2).to_string()}")

    print(f"\n  Rows with ANY missing value:")
    for name, df in [("D48", d48), ("D49", d49), ("test", test)]:
        n_miss = df.isnull().any(axis=1).sum()
        print(f"    {name}: {pct(n_miss, len(df))}")

    save_csv(miss_df, "missing_analysis.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — TARGET DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────

def section3_target(d48, d49):
    print(f"\n{DIVIDER}")
    print("  SECTION 3 — Target (demand) Distribution")
    print(DIVIDER)

    print(f"\n  Day 48 demand stats:")
    print(fmt_stats(d48["demand"]))
    print(f"\n  Day 49 demand stats:")
    print(fmt_stats(d49["demand"]))

    print(f"\n  Zero-demand rows:")
    print(f"    Day 48: {pct((d48['demand']==0).sum(), len(d48))}")
    print(f"    Day 49: {pct((d49['demand']==0).sum(), len(d49))}")

    print(f"\n  Demand > 0.5 rows:")
    print(f"    Day 48: {pct((d48['demand']>0.5).sum(), len(d48))}")
    print(f"    Day 49: {pct((d49['demand']>0.5).sum(), len(d49))}")

    print(f"\n  Demand > 1.0 rows (anomaly check):")
    print(f"    Day 48: {pct((d48['demand']>1.0).sum(), len(d48))}")
    print(f"    Day 49: {pct((d49['demand']>1.0).sum(), len(d49))}")

    bins   = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 999]
    labels = ["0-0.01","0.01-0.05","0.05-0.1","0.1-0.2",
              "0.2-0.3","0.3-0.5","0.5-1.0",">1.0"]

    print(f"\n  Demand bucket distribution (Day 48):")
    d48c = d48.copy()
    d48c["_bin"] = pd.cut(d48c["demand"], bins=bins, labels=labels, right=False)
    for label, cnt in d48c["_bin"].value_counts(sort=False).items():
        print(f"    {label:<12} {pct(cnt, len(d48))}")

    print(f"\n  Demand bucket distribution (Day 49):")
    d49c = d49.copy()
    d49c["_bin"] = pd.cut(d49c["demand"], bins=bins, labels=labels, right=False)
    for label, cnt in d49c["_bin"].value_counts(sort=False).items():
        print(f"    {label:<12} {pct(cnt, len(d49))}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — TEMPORAL COVERAGE
# ─────────────────────────────────────────────────────────────────────

def section4_temporal(d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 4 — Temporal Coverage")
    print(DIVIDER)

    for name, df in [("Day 48", d48), ("Day 49 train", d49), ("Test", test)]:
        slots = sorted(df["time_slot"].unique())
        hrs   = sorted(df["hour"].unique())
        ts    = sorted(df["timestamp"].unique())
        first_ts = f"{slots[0]//4}:{(slots[0]%4)*15:02d}"
        last_ts  = f"{slots[-1]//4}:{(slots[-1]%4)*15:02d}"
        print(f"\n  [{name}]")
        print(f"    time_slot range : {slots[0]} - {slots[-1]}  "
              f"({len(slots)} distinct slots,  {first_ts} to {last_ts})")
        print(f"    hours           : {hrs}")
        print(f"    timestamps      : {ts}")

    d48_ts  = set(d48["timestamp"].unique())
    test_ts = set(test["timestamp"].unique())
    d49_ts  = set(d49["timestamp"].unique())

    only_in_test = sorted(test_ts - d48_ts)
    print(f"\n  Test timestamps NOT in Day 48 : {len(only_in_test)} -> {only_in_test}")
    print(f"  Test timestamps NOT in Day 49 : {len(test_ts - d49_ts)}")
    print(f"  Day49 timestamps overlap Test : {len(d49_ts & test_ts)}")

    print(f"\n  Rows per timestamp (mean):")
    print(f"    Day 48      : {len(d48)/d48['timestamp'].nunique():.1f}")
    print(f"    Day 49 train: {len(d49)/d49['timestamp'].nunique():.1f}")
    print(f"    Test        : {len(test)/test['timestamp'].nunique():.1f}")

    slot_stats = []
    all_slots  = sorted(set(list(d48["time_slot"].unique()) +
                            list(d49["time_slot"].unique()) +
                            list(test["time_slot"].unique())))
    for slot in all_slots:
        d48s = d48[d48["time_slot"] == slot]
        d49s = d49[d49["time_slot"] == slot]
        ts   = test[test["time_slot"] == slot]
        slot_stats.append({
            "time_slot"       : slot,
            "time_label"      : f"{slot//4}:{(slot%4)*15:02d}",
            "in_d48_train"    : len(d48s) > 0,
            "in_d49_train"    : len(d49s) > 0,
            "in_test"         : len(ts)   > 0,
            "d48_rows"        : len(d48s),
            "d48_demand_mean" : round(d48s["demand"].mean(), 5) if len(d48s) > 0 else None,
            "d49_rows"        : len(d49s),
            "d49_demand_mean" : round(d49s["demand"].mean(), 5) if len(d49s) > 0 else None,
            "test_rows"       : len(ts),
        })
    slot_df = pd.DataFrame(slot_stats)
    print(f"\n  Per-slot presence table:")
    print(slot_df.to_string(index=False))
    save_csv(slot_df, "temporal_slot_stats.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — GEOHASH COVERAGE & OVERLAP
# ─────────────────────────────────────────────────────────────────────

def section5_geohash(d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 5 — Geohash Coverage & Overlap")
    print(DIVIDER)

    gh_d48  = set(d48["geohash"].unique())
    gh_d49  = set(d49["geohash"].unique())
    gh_test = set(test["geohash"].unique())

    print(f"\n  Unique geohashes:")
    print(f"    Day 48       : {len(gh_d48)}")
    print(f"    Day 49 train : {len(gh_d49)}")
    print(f"    Test         : {len(gh_test)}")

    print(f"\n  Test geohash overlap:")
    print(f"    Test in Day 48       : {len(gh_test & gh_d48)}  "
          f"({100*len(gh_test & gh_d48)/len(gh_test):.1f}%)")
    print(f"    Test in Day 49 train : {len(gh_test & gh_d49)}  "
          f"({100*len(gh_test & gh_d49)/len(gh_test):.1f}%)")
    print(f"    Test in BOTH D48+D49 : {len(gh_test & gh_d48 & gh_d49)}")
    print(f"    Test NOT in Day 48   : {len(gh_test - gh_d48)}")
    print(f"    Test NOT in Day 49   : {len(gh_test - gh_d49)}")
    print(f"    Test NOT in either   : {len(gh_test - gh_d48 - gh_d49)}  <- cold-start")

    for prefix_len in [3, 4, 5]:
        col = f"geo_p{prefix_len}"
        test_pre = set(test["geohash"].str[:prefix_len].unique())
        d48_pre  = set(d48["geohash"].str[:prefix_len].unique())
        overlap  = 100 * len(test_pre & d48_pre) / len(test_pre)
        print(f"    {col} test coverage in Day48: "
              f"{len(test_pre & d48_pre)}/{len(test_pre)} ({overlap:.1f}%)")

    geo_stats = []
    for gh in sorted(gh_test):
        in_d48 = gh in gh_d48
        in_d49 = gh in gh_d49
        d48r   = d48[d48["geohash"] == gh]
        d49r   = d49[d49["geohash"] == gh]
        geo_stats.append({
            "geohash"        : gh,
            "in_d48_train"   : in_d48,
            "in_d49_train"   : in_d49,
            "d48_n_slots"    : len(d48r),
            "d48_mean_demand": round(d48r["demand"].mean(), 5) if in_d48 else None,
            "d48_std_demand" : round(d48r["demand"].std(),  5) if in_d48 else None,
            "d49_n_slots"    : len(d49r),
            "d49_mean_demand": round(d49r["demand"].mean(), 5) if in_d49 else None,
        })
    geo_df = pd.DataFrame(geo_stats)

    cold = geo_df[~geo_df["in_d48_train"]]
    print(f"\n  Cold-start geohashes (not in Day 48):")
    if len(cold) > 0:
        print(cold[["geohash", "in_d49_train", "d49_mean_demand"]].to_string(index=False))
    else:
        print("    None — all test geohashes appear in Day 48!")

    save_csv(geo_df, "geohash_overlap.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 6 — LAG QUALITY
# ─────────────────────────────────────────────────────────────────────

def section6_lag_quality(d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 6 — Lag Quality  (D48 -> D49-train and D48 -> Test)")
    print(DIVIDER)

    d48_map = d48.set_index(["geohash", "timestamp"])["demand"]

    d49c = d49.copy()
    d49c["demand_d48"] = [d48_map.get((g, t), np.nan)
                          for g, t in zip(d49c["geohash"], d49c["timestamp"])]

    testc = test.copy()
    testc["demand_d48"] = [d48_map.get((g, t), np.nan)
                           for g, t in zip(testc["geohash"], testc["timestamp"])]

    print(f"\n  Day-49 train lag coverage (exact geohash+timestamp match):")
    print(f"    matched  : {pct(d49c['demand_d48'].notna().sum(), len(d49c))}")
    print(f"    unmatched: {pct(d49c['demand_d48'].isna().sum(),  len(d49c))}")

    print(f"\n  Test lag coverage (exact geohash+timestamp match):")
    print(f"    matched  : {pct(testc['demand_d48'].notna().sum(), len(testc))}")
    print(f"    unmatched: {pct(testc['demand_d48'].isna().sum(),  len(testc))}")

    matched = d49c.dropna(subset=["demand_d48"])
    if len(matched) > 0:
        from sklearn.metrics import r2_score
        r2   = r2_score(matched["demand"], matched["demand_d48"])
        corr = matched[["demand", "demand_d48"]].corr().iloc[0, 1]
        rmse = np.sqrt(np.mean((matched["demand"] - matched["demand_d48"])**2))
        mae  = (matched["demand"] - matched["demand_d48"]).abs().mean()
        print(f"\n  Lag as predictor of Day-49 (matched rows):")
        print(f"    n_rows  = {len(matched)}")
        print(f"    R2      = {r2:.4f}  (score = {100*max(0,r2):.2f})")
        print(f"    Corr    = {corr:.4f}")
        print(f"    RMSE    = {rmse:.5f}")
        print(f"    MAE     = {mae:.5f}")
        print(f"    Max err = {(matched['demand'] - matched['demand_d48']).abs().max():.5f}")

    # Per-slot lag quality
    from sklearn.metrics import r2_score as r2s
    slot_lag = []
    for slot, grp in matched.groupby("time_slot"):
        if len(grp) < 3:
            continue
        r2 = max(0, r2s(grp["demand"], grp["demand_d48"]))
        slot_lag.append({
            "time_slot"   : slot,
            "time_label"  : f"{slot//4}:{(slot%4)*15:02d}",
            "n"           : len(grp),
            "r2_lag"      : round(r2, 4),
            "demand_mean" : round(grp["demand"].mean(), 5),
            "d48_mean"    : round(grp["demand_d48"].mean(), 5),
        })
    slot_lag_df = pd.DataFrame(slot_lag)
    if len(slot_lag_df) > 0:
        print(f"\n  Lag R2 by time slot (Day-49 train matched rows):")
        print(slot_lag_df.to_string(index=False))
        save_csv(slot_lag_df, "lag_quality_by_slot.csv")

    # Per-geohash lag coverage for test
    gh_lag = (testc.groupby("geohash")["demand_d48"]
              .apply(lambda x: x.notna().mean())
              .reset_index()
              .rename(columns={"demand_d48": "lag_coverage"}))
    save_csv(gh_lag, "test_lag_coverage_by_geohash.csv")
    print(f"\n  Test geohashes by lag coverage bucket:")
    print(f"    Full  (100%) : {(gh_lag['lag_coverage']==1.0).sum()}")
    print(f"    Partial      : {((gh_lag['lag_coverage']>0) & (gh_lag['lag_coverage']<1)).sum()}")
    print(f"    None  (0%)   : {(gh_lag['lag_coverage']==0.0).sum()}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 7 — DAY-48 to DAY-49 DRIFT
# ─────────────────────────────────────────────────────────────────────

def section7_drift(d48, d49):
    print(f"\n{DIVIDER}")
    print("  SECTION 7 — Day-48 -> Day-49 Drift")
    print(DIVIDER)

    d48_map = d48.set_index(["geohash", "timestamp"])["demand"]
    d49c    = d49.copy()
    d49c["demand_d48"] = [d48_map.get((g, t), np.nan)
                          for g, t in zip(d49c["geohash"], d49c["timestamp"])]
    matched = d49c.dropna(subset=["demand_d48"])

    global_drift = matched["demand"].mean() / (matched["demand_d48"].mean() + 1e-9)
    row_ratios   = matched["demand"] / (matched["demand_d48"] + 1e-9)
    print(f"\n  Global drift (D49_mean / D48_mean) : {global_drift:.4f}")
    print(f"  Median row-level ratio (D49/D48)   : {row_ratios.median():.4f}")
    print(f"  Std of row-level ratio             : {row_ratios.std():.4f}")
    print(f"  Row ratio range                    : {row_ratios.min():.4f} - {row_ratios.max():.4f}")

    # Per-geohash drift
    def geo_drift(g):
        return pd.Series({
            "n"        : len(g),
            "d49_mean" : g["demand"].mean(),
            "d48_mean" : g["demand_d48"].mean(),
            "drift"    : g["demand"].mean() / (g["demand_d48"].mean() + 1e-9),
            "corr"     : g[["demand","demand_d48"]].corr().iloc[0,1]
        })

    gh_drift = matched.groupby("geohash").apply(geo_drift).reset_index()
    gh_drift["drift"] = gh_drift["drift"].clip(0.05, 20)

    print(f"\n  Per-geohash drift stats:")
    print(fmt_stats(gh_drift["drift"]))
    print(f"\n  Top 5 highest drift geohashes:")
    print(gh_drift.nlargest(5, "drift")[
        ["geohash","n","d49_mean","d48_mean","drift"]].round(5).to_string(index=False))
    print(f"\n  Top 5 lowest drift geohashes:")
    print(gh_drift.nsmallest(5, "drift")[
        ["geohash","n","d49_mean","d48_mean","drift"]].round(5).to_string(index=False))

    # Per-slot drift
    def slot_drift(g):
        return pd.Series({
            "n"        : len(g),
            "d49_mean" : g["demand"].mean(),
            "d48_mean" : g["demand_d48"].mean(),
            "drift"    : g["demand"].mean() / (g["demand_d48"].mean() + 1e-9)
        })

    slot_drift_df = matched.groupby("time_slot").apply(slot_drift).reset_index()
    slot_drift_df["time_label"] = (slot_drift_df["time_slot"]//4).astype(str) + ":" + \
        ((slot_drift_df["time_slot"]%4)*15).apply(lambda m: f"{m:02d}")
    print(f"\n  Per-time-slot drift:")
    print(slot_drift_df.round(5).to_string(index=False))

    save_csv(gh_drift,        "drift_by_geohash.csv")
    save_csv(slot_drift_df,   "drift_by_slot.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 8 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def section8_features(train_raw, d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 8 — Feature Analysis")
    print(DIVIDER)

    cat_feats = ["RoadType", "Weather", "NumberofLanes",
                 "LargeVehicles", "Landmarks"]

    for col in cat_feats:
        print(f"\n  [{col}]")
        all_vals = pd.concat([train_raw[col], test[col]], ignore_index=True)
        print(f"    Unique values  : {sorted(all_vals.dropna().unique())}")
        print(f"    Missing D48    : {pct(d48[col].isna().sum(),  len(d48))}")
        print(f"    Missing D49    : {pct(d49[col].isna().sum(),  len(d49))}")
        print(f"    Missing test   : {pct(test[col].isna().sum(), len(test))}")
        print(f"    D48 distribution:")
        for v, c in d48[col].value_counts(dropna=False).items():
            print(f"      {str(v):<25} {pct(c, len(d48))}")
        print(f"    Test distribution:")
        for v, c in test[col].value_counts(dropna=False).items():
            print(f"      {str(v):<25} {pct(c, len(test))}")

    print(f"\n  [Temperature]")
    print(f"    D48 stats  : {fmt_stats(d48['Temperature'])}")
    print(f"    D49 stats  : {fmt_stats(d49['Temperature'])}")
    print(f"    Test stats : {fmt_stats(test['Temperature'])}")
    print(f"    Missing D48 : {pct(d48['Temperature'].isna().sum(),  len(d48))}")
    print(f"    Missing D49 : {pct(d49['Temperature'].isna().sum(),  len(d49))}")
    print(f"    Missing test: {pct(test['Temperature'].isna().sum(), len(test))}")

    print(f"\n  [Feature consistency per geohash in train_raw]")
    for col in ["RoadType", "Weather", "NumberofLanes", "LargeVehicles", "Landmarks"]:
        n_unique = (train_raw.dropna(subset=[col])
                    .groupby("geohash")[col].nunique())
        inconsistent = (n_unique > 1).sum()
        note = "INCONSISTENT -> treat as soft/noisy signal" if inconsistent > 0 else "consistent"
        print(f"    {col:<20}: {inconsistent} geohashes with >1 value  ({note})")

    print(f"\n  [Demand by RoadType  (Day 48)]")
    rt = d48.groupby("RoadType")["demand"].agg(["mean","std","count"])
    print(rt.round(5).to_string())

    print(f"\n  [Demand by Weather  (Day 48)]")
    wt = d48.groupby("Weather")["demand"].agg(["mean","std","count"])
    print(wt.round(5).to_string())

    print(f"\n  [Demand by NumberofLanes  (Day 48)]")
    nl = d48.groupby("NumberofLanes")["demand"].agg(["mean","std","count"])
    print(nl.round(5).to_string())

    print(f"\n  [Demand by LargeVehicles  (Day 48)]")
    lv = d48.groupby("LargeVehicles")["demand"].agg(["mean","std","count"])
    print(lv.round(5).to_string())

    print(f"\n  [Demand by Landmarks  (Day 48)]")
    lm = d48.groupby("Landmarks")["demand"].agg(["mean","std","count"])
    print(lm.round(5).to_string())

    d48_notna = d48.dropna(subset=["Temperature"])
    corr_temp = d48_notna[["Temperature","demand"]].corr().iloc[0,1]
    print(f"\n  Pearson correlation(Temperature, demand) in D48: {corr_temp:.4f}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 9 — DEMAND DISTRIBUTIONS BY GROUP
# ─────────────────────────────────────────────────────────────────────

def section9_demand_groups(d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 9 — Demand Distributions by Group")
    print(DIVIDER)

    print(f"\n  Demand by hour (Day 48):")
    hourly = d48.groupby("hour")["demand"].agg(["mean","std","count"]).round(5)
    print(hourly.to_string())

    slot_stats = (d48.groupby("time_slot")["demand"]
                  .agg(["mean","std","count"])
                  .reset_index())
    slot_stats["time_label"] = (slot_stats["time_slot"]//4).astype(str) + ":" + \
        ((slot_stats["time_slot"]%4)*15).apply(lambda m: f"{m:02d}")
    print(f"\n  Demand by time_slot (Day 48)  [full 96-slot table]:")
    print(slot_stats.round(5).to_string(index=False))
    save_csv(slot_stats, "demand_by_slot_d48.csv")

    print(f"\n  Top-10 peak demand slots (Day 48):")
    print(slot_stats.nlargest(10, "mean")[
        ["time_slot","time_label","mean","std"]].to_string(index=False))

    print(f"\n  Bottom-10 demand slots (Day 48):")
    print(slot_stats.nsmallest(10, "mean")[
        ["time_slot","time_label","mean","std"]].to_string(index=False))

    if len(d49) > 0:
        print(f"\n  Demand by hour (Day 49 train - early window):")
        hourly_d49 = d49.groupby("hour")["demand"].agg(["mean","std","count"]).round(5)
        print(hourly_d49.to_string())

    geo_stats = (d48.groupby("geohash")["demand"]
                 .agg(["mean","std","count"])
                 .sort_values("mean", ascending=False)
                 .reset_index())
    print(f"\n  Top 20 geohashes by mean demand (Day 48):")
    print(geo_stats.head(20).round(5).to_string(index=False))
    save_csv(geo_stats, "geohash_demand_stats_d48.csv")


# ─────────────────────────────────────────────────────────────────────
# SECTION 10 — TEST SET CHARACTERISTICS
# ─────────────────────────────────────────────────────────────────────

def section10_test(d48, d49, test, sample_sub):
    print(f"\n{DIVIDER}")
    print("  SECTION 10 — Test Set Characteristics")
    print(DIVIDER)

    gh_d48  = set(d48["geohash"].unique())
    gh_d49  = set(d49["geohash"].unique())
    gh_test = set(test["geohash"].unique())

    rows_per_gh = test["geohash"].value_counts()
    print(f"\n  Rows per geohash in test:")
    print(f"    min={rows_per_gh.min()}  max={rows_per_gh.max()}  "
          f"mean={rows_per_gh.mean():.1f}  "
          f"all_same={(rows_per_gh.nunique()==1)}")
    if rows_per_gh.nunique() > 1:
        print("    Geohashes with non-standard row count:")
        mode_count = rows_per_gh.mode()[0]
        print(rows_per_gh[rows_per_gh != mode_count].to_string())

    cold_start = gh_test - gh_d48 - gh_d49
    print(f"\n  Cold-start geohashes (not in D48 or D49 train): {len(cold_start)}")
    if cold_start:
        print(f"    {sorted(cold_start)}")

    print(f"\n  Sample submission demand stats:")
    print(fmt_stats(sample_sub["demand"]))

    print(f"\n  Test missing values per column:")
    for col in test.columns:
        miss = test[col].isna().sum()
        if miss > 0:
            print(f"    {col:<20}: {pct(miss, len(test))}")

    print(f"\n  RoadType distribution comparison (test vs D48):")
    t_rt = test["RoadType"].value_counts(normalize=True, dropna=False).round(3)
    d_rt = d48["RoadType"].value_counts(normalize=True, dropna=False).round(3)
    comp = pd.DataFrame({"test_pct": t_rt, "d48_pct": d_rt}).fillna(0)
    print(comp.to_string())

    print(f"\n  Weather distribution comparison (test vs D48):")
    t_wt = test["Weather"].value_counts(normalize=True, dropna=False).round(3)
    d_wt = d48["Weather"].value_counts(normalize=True, dropna=False).round(3)
    comp2 = pd.DataFrame({"test_pct": t_wt, "d48_pct": d_wt}).fillna(0)
    print(comp2.to_string())


# ─────────────────────────────────────────────────────────────────────
# SECTION 11 — KEY FINDINGS SUMMARY
# ─────────────────────────────────────────────────────────────────────

def section11_summary(d48, d49, test):
    print(f"\n{DIVIDER}")
    print("  SECTION 11 — Key Findings & Model Hints")
    print(DIVIDER)

    gh_d48  = set(d48["geohash"].unique())
    gh_d49  = set(d49["geohash"].unique())
    gh_test = set(test["geohash"].unique())
    cold    = len(gh_test - gh_d48 - gh_d49)

    d48_map = d48.set_index(["geohash", "timestamp"])["demand"]
    testc   = test.copy()
    testc["lag_ok"] = [
        (g, t) in d48_map.index
        for g, t in zip(testc["geohash"], testc["timestamp"])
    ]
    lag_cov = testc["lag_ok"].mean()

    d49_slots = sorted(d49["time_slot"].unique())
    test_slots = sorted(test["time_slot"].unique())
    overlap_slots = set(d49_slots) & set(test_slots)

    print(f"""
  DATA PROFILE
  {SECTION}
  Train : Day 48 ({len(d48)} rows, {len(gh_d48)} geohashes, all 96 time slots)
          Day 49 ({len(d49)} rows, {d49["geohash"].nunique()} geohashes,
                  slots {min(d49_slots)}-{max(d49_slots)}  i.e. {min(d49_slots)//4}:{(min(d49_slots)%4)*15:02d} - {max(d49_slots)//4}:{(max(d49_slots)%4)*15:02d})
  Test  : Day 49 ({len(test)} rows, {len(gh_test)} geohashes,
                  slots {min(test_slots)}-{max(test_slots)}  i.e. {min(test_slots)//4}:{(min(test_slots)%4)*15:02d} - {max(test_slots)//4}:{(max(test_slots)%4)*15:02d})

  TEMPORAL SPLIT INSIGHT
  {SECTION}
  - D49 train covers slots {min(d49_slots)}-{max(d49_slots)} (early morning)
  - Test   covers slots {min(test_slots)}-{max(test_slots)} (rest of day)
  - Overlap between D49-train timestamps and test timestamps: {len(overlap_slots)} slots
  - This means D49-train and test are NON-OVERLAPPING time windows.
  - d49_d48_ratio from D49-train is 100% leak-free for test predictions.
  - But it IS leaky for OOF when D49-train rows are used as validation set.

  CRITICAL MODEL DESIGN NOTES
  {SECTION}
  1. LAG COVERAGE
     Exact geohash+timestamp match for test: {lag_cov*100:.1f}%
     -> demand_d48 is the #1 feature. Preserve it at all costs.
     -> The remaining {(1-lag_cov)*100:.1f}% needs a robust fallback chain.

  2. COLD-START
     Test geohashes not in any training day: {cold}
     -> Need geo-prefix fallback (p5 -> p4 -> p3) for these.

  3. AGGREGATE STATS SOURCE
     -> Use ONLY Day 48 as stats_src in add_aggregate_stats().
     -> Day 49 demand must NEVER feed into OOF training features.

  4. TRAINING DATA STRATEGY
     -> Training on both Day 48 + Day 49 gives more data.
     -> But sample_weight=3-5 for Day 49 rows helps since test is Day 49.
     -> Check Section 8: if RoadType/Weather are inconsistent per geohash,
        treat them as weak/noisy features, not hard rules.

  5. DRIFT FEATURE
     -> d49_d48_ratio per geohash (from D49-train) is a powerful, leak-free
        feature for test. Use it as: demand_d48_corrected = demand_d48 * ratio.
     -> Check Section 7 output to see if drift is stable or volatile.

  Output CSVs saved to: outputs/eda/
  {SECTION}
""")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print(DIVIDER)
    print("  GRIDLOCK HACKATHON 2.0 — Comprehensive EDA")
    print(DIVIDER)

    train_raw, d48, d49, test, sample_sub = load()

    section1_shape(train_raw, d48, d49, test, sample_sub)
    section2_missing(train_raw, d48, d49, test)
    section3_target(d48, d49)
    section4_temporal(d48, d49, test)
    section5_geohash(d48, d49, test)
    section6_lag_quality(d48, d49, test)
    section7_drift(d48, d49)
    section8_features(train_raw, d48, d49, test)
    section9_demand_groups(d48, d49, test)
    section10_test(d48, d49, test, sample_sub)
    section11_summary(d48, d49, test)

    print(f"\n{DIVIDER}")
    print("  EDA complete.  All CSVs saved to outputs/eda/")
    print(DIVIDER)


if __name__ == "__main__":
    main()
