# Gridlock Hackathon 2.0 — Traffic Demand Prediction
## Complete Solution Guide

---

## Folder Structure

```
gridlock_solution/
│
├── data/                   ← PUT YOUR DATA FILES HERE
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/                 ← Auto-created — saved model files
├── outputs/                ← Auto-created — submission.csv goes here
├── logs/                   ← Auto-created — optional logs
│
├── train.py                ← MAIN script (run this)
├── eda.py                  ← Optional: explore data first
├── feature_importance.py   ← Optional: view top features after training
├── requirements.txt
└── README.md
```

---

## Step-by-Step: How to Train

### Step 0 — Setup (one time only)

```bash
# Go into the project folder
cd gridlock_solution

# Install dependencies
pip install -r requirements.txt
```

---

### Step 1 — Add Your Data

Download the dataset from HackerEarth and place all 3 files inside the `data/` folder:

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

---

### Step 2 — (Optional) Run EDA First

This checks if the Day-48 lag feature will work well on your data.

```bash
python eda.py
```

**What to look for:**
- `Day-48 lag coverage` — if it says [HIGH] (>80%), the lag feature is your golden predictor
- Any missing values in key columns
- What days are in train vs test

---

### Step 3 — Train the Model

```bash
python train.py
```

**What happens:**
1. Loads train.csv and test.csv
2. Engineers 30+ features (temporal, geo, lag, target encoding, aggregates)
3. Trains LightGBM with 5-fold cross-validation
4. Prints fold scores in real-time
5. Saves models to `models/lgbm_models.pkl`
6. Saves predictions to `outputs/submission.csv`

**Expected output:**
```
=======================================================
  GRIDLOCK HACKATHON 2.0 — Traffic Demand Prediction
=======================================================
[1/6] Loading data...
  Train: (77299, 11)  |  Test: (41778, 10)

[2/6] Feature engineering...
[3/6] Building lag features...
  Day-lag coverage on test: 98.7%
[4/6] Target encoding (OOF)...
[5/6] Aggregate statistics...
[6/6] Training LightGBM (5-fold CV)...
  Features: 38

  Fold 1 | Score: 96.21 | Trees: 847 | Time: 42.3s
  Fold 2 | Score: 95.87 | Trees: 912 | Time: 38.1s
  Fold 3 | Score: 96.44 | Trees: 788 | Time: 40.2s
  Fold 4 | Score: 95.93 | Trees: 903 | Time: 41.0s
  Fold 5 | Score: 96.11 | Trees: 856 | Time: 39.5s

  ── OOF Score: 96.11 ──

  Models saved → models/lgbm_models.pkl
  Submission saved → outputs/submission.csv  shape: (41778, 2)
  Total time: 3.4 min
=======================================================
```

**Expected score range:** 93–99 depending on lag coverage

---

### Step 4 — (Optional) Check Feature Importance

```bash
python feature_importance.py
```

This shows which features mattered most. Typically:
1. `demand_d48` (lag feature) — highest by far if coverage is good
2. `geohash_x_time_slot_enc` (interaction target encoding)
3. `geohash_mean` / `geohash_median` (location-level aggregates)

---

### Step 5 — Submit

1. Go to the HackerEarth problem page
2. Under **Upload Prediction File** → choose `outputs/submission.csv`
3. Under **Upload Source Code** → zip your `.ipynb` or this folder and upload
4. Click **Submit & Evaluate**

---

## Key Ideas in This Solution

| Feature | Why It Helps |
|---|---|
| `demand_d48` | Test data is Day 49; Day 48 same location+time is nearly identical |
| `geohash × time_slot encoding` | Captures rush-hour patterns specific to each location |
| Cyclical sin/cos time | Midnight and 23:45 are "close" — avoids discontinuity |
| Geohash prefixes (p3–p6) | Hierarchical spatial grouping for fallback encoding |
| Per-location aggregates | Baseline demand level at each location |

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: lightgbm` | `pip install lightgbm` |
| `FileNotFoundError: data/train.csv` | Make sure files are in the `data/` folder |
| Low score (<85) | Check `eda.py` output -- lag coverage might be low |
| OOM / slow | Reduce `n_estimators` to 1000 in `train.py` LGBM_PARAMS |
| pygeohash missing | Install it: `pip install pygeohash` (optional, not required) |