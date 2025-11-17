import pandas as pd
from xgboost import XGBRegressor
import os

# 1. Load dataset
df = pd.read_csv("monthly-car-sales.csv")

# Rename column for convenience
df.columns = ["Month", "Sales"]

# 2. Create lag features for last 6 months
for i in range(1, 7):
    df[f"lag_{i}"] = df["Sales"].shift(i)

# Drop rows with NaN (first 6 months)
df = df.dropna()

# 3. Prepare data
feature_cols = [f"lag_{i}" for i in range(1, 7)]
X = df[feature_cols]
y = df["Sales"]

# 4. Train XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X, y)

# 5. Save model as JSON (Booster)
os.makedirs("model", exist_ok=True)
booster = model.get_booster()
booster.save_model("model/model.json")

print("✅ Model saved to model/model.json")
