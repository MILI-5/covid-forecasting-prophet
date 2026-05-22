import os
import joblib
import numpy as np

from src.models.prophet_model import (
    load_prophet_data,
    visualize_data,
    train_prophet_model,
    make_forecast,
    plot_forecast,
    plot_actual_vs_forecast,
    train_on_train_set,
    evaluate_on_test
)

# =========================================================
# Step 1 — Load Data
# =========================================================
df = load_prophet_data()

# =========================================================
# Step 2 — Visual Check
# =========================================================
visualize_data(df)

# =========================================================
# Step 3–5 — Full Model Training
# =========================================================
model_full = train_prophet_model(df)

# =========================================================
# Step 6–8 — Forecast
# =========================================================
forecast = make_forecast(model_full, periods=30)

# =========================================================
# Step 9 — Visualization
# =========================================================
plot_forecast(model_full, forecast)

# =========================================================
# Step 10 — Actual vs Predicted
# =========================================================
plot_actual_vs_forecast(df, forecast)

# =========================================================
# Train/Test Split
# =========================================================
train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

# =========================================================
# Step 13 — Train on Train Set Only
# =========================================================
model_eval = train_on_train_set(train)

# =========================================================
# Step 14 — Predict on Test Set
# =========================================================
forecast_test = evaluate_on_test(model_eval, test)

# =========================================================
# Step 15 — Align Predictions
# =========================================================
predicted = forecast_test.iloc[-len(test):]["yhat"]
actual = test["y"].values

# Optional: evaluation metric example
mae = np.mean(np.abs(actual - predicted))

print("\n📊 Evaluation Results")
print("MAE:", mae)

# =========================================================
# Step 16 — Save Model (Production Step)
# =========================================================
os.makedirs("outputs/model", exist_ok=True)

joblib.dump(model_eval, "outputs/model/prophet_model.pkl")

print("\n✅ Model saved at outputs/model/prophet_model.pkl")