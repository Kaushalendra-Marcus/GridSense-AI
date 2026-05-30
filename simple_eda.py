import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# LOAD DATA
# =====================================================

TRAIN_PATH = "data/train.csv"

df = pd.read_csv(TRAIN_PATH)

print("=" * 80)
print("DATASET SHAPE")
print("=" * 80)
print(df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nINFO")
print(df.info())

# =====================================================
# MISSING VALUES
# =====================================================

print("\nMISSING VALUES")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

plt.figure(figsize=(10,5))
missing[missing > 0].plot(kind="bar")
plt.title("Missing Values")
plt.tight_layout()
plt.show()

# =====================================================
# DUPLICATES
# =====================================================

print("\nDUPLICATES")
print(df.duplicated().sum())

# =====================================================
# TARGET ANALYSIS
# =====================================================

TARGET = "demand"

print("\nTARGET DESCRIPTION")
print(df[TARGET].describe())

plt.figure(figsize=(8,5))
sns.histplot(df[TARGET], kde=True)
plt.title("Target Distribution")
plt.show()

# =====================================================
# NUMERIC FEATURES
# =====================================================

numeric_cols = df.select_dtypes(
    include=["int64","float64"]
).columns.tolist()

if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print("\nNUMERIC FEATURES")
print(numeric_cols)

# =====================================================
# CATEGORICAL FEATURES
# =====================================================

categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCATEGORICAL FEATURES")
print(categorical_cols)

# =====================================================
# CARDINALITY
# =====================================================

print("\nCATEGORICAL CARDINALITY")

for col in categorical_cols:
    print(f"{col}: {df[col].nunique()}")

# =====================================================
# CORRELATION
# =====================================================

corr_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))
sns.heatmap(
    corr_df.corr(),
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Matrix")
plt.show()

# =====================================================
# CORRELATION WITH TARGET
# =====================================================

print("\nCORRELATION WITH TARGET")

target_corr = (
    corr_df.corr()[TARGET]
    .sort_values(ascending=False)
)

print(target_corr)

# =====================================================
# OUTLIER CHECK
# =====================================================

for col in numeric_cols[:10]:

    plt.figure(figsize=(6,2))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# =====================================================
# TIMESTAMP ANALYSIS
# =====================================================

if "timestamp" in df.columns:

    print("\nTIMESTAMP DETECTED")

    try:

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        print(df[["hour","dayofweek","month"]].head())

        plt.figure(figsize=(10,5))
        df.groupby("hour")[TARGET].mean().plot()
        plt.title("Demand by Hour")
        plt.show()

        plt.figure(figsize=(10,5))
        df.groupby("dayofweek")[TARGET].mean().plot()
        plt.title("Demand by Day Of Week")
        plt.show()

    except:
        print("Timestamp parsing failed")

# =====================================================
# CATEGORICAL VS TARGET
# =====================================================

for col in categorical_cols:

    if df[col].nunique() <= 20:

        plt.figure(figsize=(10,5))

        df.groupby(col)[TARGET] \
            .mean() \
            .sort_values() \
            .plot(kind="bar")

        plt.title(f"{col} vs Demand")
        plt.tight_layout()
        plt.show()

# =====================================================
# BASELINE FEATURE IMPORTANCE
# =====================================================

print("\nTRAINING BASELINE RANDOM FOREST")

temp = df.copy()

for col in temp.columns:

    if temp[col].dtype == "object":

        le = LabelEncoder()

        temp[col] = le.fit_transform(
            temp[col].astype(str)
        )

temp = temp.fillna(-999)

X = temp.drop(columns=[TARGET])
y = temp[TARGET]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTOP FEATURES")
print(importance.head(20))

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance.head(20),
    x="importance",
    y="feature"
)
plt.title("Top 20 Important Features")
plt.show()

# =====================================================
# SAVE REPORT
# =====================================================

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nSaved: feature_importance.csv")
print("EDA Complete")