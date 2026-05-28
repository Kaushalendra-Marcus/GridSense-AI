# Traffic Demand Prediction

A LightGBM-based solution for predicting normalized traffic demand across locations and time slots, built for the Gridlock Hackathon 2.0 (Flipkart).

OOF R2 Score: **95.63 / 100**

---

## Project Structure

```
gridlock_solution/
├── data/                    # Input data files (not tracked)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                  # Saved model artifacts (auto-created)
├── outputs/                 # Prediction files (auto-created)
├── train.py                 # Main training pipeline
├── eda.py                   # Data exploration and lag coverage check
├── feature_importance.py    # Feature importance from saved models
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset files in the `data/` folder before running anything.

---

## Usage

### 1. Explore the data (optional)

```bash
python eda.py
```

Prints dataset statistics, missing value counts, day/timestamp distribution, geohash coverage, and Day-48 lag overlap between train and test.

### 2. Train

```bash
python train.py
```

Runs the full pipeline and writes predictions to `outputs/submission.csv`.

### 3. Inspect feature importance (optional)

```bash
python feature_importance.py
```

Prints and saves a ranked feature importance table from the trained models.

---

## Pipeline Overview

The pipeline runs in 6 stages:

**1. Load** — reads `train.csv` and `test.csv`.

**2. Feature engineering** — extracts temporal features (hour, minute, 15-minute time slot, peak/night flags, cyclical sin/cos encodings for hour, time slot, and day), spatial features (geohash prefix hierarchy at levels 3-6, decoded lat/lon), and fills missing values in `RoadType`, `Weather`, and `Temperature` using geo-prefix mode and median fallbacks.

**3. Lag feature** — joins Day 48 demand onto Day 49 train rows and test rows by `geohash + timestamp`. Rows with no match are filled using geo-prefix + timestamp median, then timestamp median, then global median. Final NaN count is zero.

**4. Target encoding (OOF)** -- encodes 9 group keys using out-of-fold means to prevent leakage: `geohash`, `geo_p4`, `geohash x time_slot`, `geohash x day`, `geo_p4 x time_slot`, `day x time_slot`, `RoadType x time_slot`, `geohash x is_peak`, `Weather x time_slot`.

**5. Aggregate statistics** -- computes per-group demand statistics (mean, std, median, max, min) for `geohash`, `time_slot`, `geo_p4`, `RoadType`, `NumberofLanes`, and `Weather`. Also derives `lag / geo_mean ratio`, `lag - geo_median`, and a coarse temperature bin.

**6. Train** -- fits LightGBM with 5-fold cross-validation. Predictions are averaged across folds. Models and feature columns are saved to `models/lgbm_models.pkl`.

---

## Features (49 total)

| Group | Features |
|---|---|
| Temporal | `time_slot`, `is_peak`, `is_night`, `sin/cos` encodings for hour, slot, day |
| Spatial | `geo_p3/4/5/6`, `lat`, `lon` |
| Lag | `demand_d48`, `lag_to_geo_mean_ratio`, `lag_minus_geo_median` |
| Target encoded | 9 interaction encodings (OOF) |
| Aggregate stats | 13 group-level demand statistics |
| Raw | `day`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`, `temp_bin` |

---

## Model

LightGBM regressor, 5-fold CV, optimized for R2 score.

```
n_estimators:      5000 (with early stopping, patience=200)
learning_rate:     0.02
num_leaves:        255
min_child_samples: 15
subsample:         0.75
colsample_bytree:  0.75
reg_alpha:         0.05
reg_lambda:        0.10
```

---

## Submission

The output file `outputs/submission.csv` contains two columns: `Index` and `demand` (41778 rows).

To submit on HackerEarth:
1. Go to the problem page and scroll to **Upload Prediction File**
2. Select `outputs/submission.csv` and click **Submit & Evaluate**
3. Under **Upload Source Code**, upload a zip of this folder

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: lightgbm` | `pip install lightgbm` |
| `FileNotFoundError: data/train.csv` | Place dataset files in the `data/` folder |
| `pygeohash` not found | `pip install pygeohash` -- optional, skipped automatically if missing |
| Slow training / OOM | Lower `n_estimators` to 2000 and `num_leaves` to 127 in `LGBM_PARAMS` |