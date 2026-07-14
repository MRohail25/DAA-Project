import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# =====================
# Load dataset
# =====================
df = pd.read_csv("data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# =====================
# Fix datetime format
# (handles both 'a.m' and 'p.m' formats)
# =====================
df["date_time"] = (
    df["date_time"]
    .astype(str)
    .str.strip()
    .str.replace("a.m", "AM", regex=False)
    .str.replace("p.m", "PM", regex=False)
)

df["date_time"] = pd.to_datetime(
    df["date_time"],
    format="%d/%m/%Y %I:%M %p"
)

# =====================
# Feature engineering
# =====================
df["hour"]  = df["date_time"].dt.hour
df["day"]   = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month

# =====================
# Features (X) and Target (y)
# NOTE: vehicle_count is the target.
# Total_count has 23/24 values missing — unusable as a target.
# =====================
X = df[["hour", "day", "month", "Average_hourly"]]
y = df["vehicle_count"]

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================
# Model training
# =====================
model = RandomForestRegressor(
    n_estimators=100,   # 200 is overkill for 24 rows
    random_state=42
)
model.fit(X_train, y_train)

# =====================
# Evaluate
# =====================
preds = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, preds):.2f}")
print(f"R²  : {r2_score(y_test, preds):.4f}")

# =====================
# Save model
# =====================
joblib.dump(model, "traffic_model.pkl")
print("✅ Model trained and saved!")