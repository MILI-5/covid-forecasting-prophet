import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics


# =========================================================
# Step 1 — Load Data
# =========================================================
def load_data():
    df = pd.read_csv("data/processed/india_covid_clean.csv")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    return df


# =========================================================
# Step 2 — Train-Test Split
# =========================================================
def train_test_split(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)

    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    return train, test


# =========================================================
# Step 3 — Train Model
# =========================================================
def train_model(train):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    model.fit(train)
    return model


# =========================================================
# Step 4 — Predict Test Period
# =========================================================
def predict_test(model, test):
    future = model.make_future_dataframe(
        periods=len(test),
        freq="D"
    )

    forecast = model.predict(future)
    return forecast


# =========================================================
# Step 5 — Extract Predictions
# =========================================================
def get_predictions(forecast, test):
    predicted = forecast["yhat"].iloc[-len(test):].values
    actual = test["y"].values

    return predicted, actual


# =========================================================
# Step 6 — Metrics
# =========================================================
def evaluate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)

    return mae, rmse, mape


# =========================================================
# Step 7 — Interpretation
# =========================================================
def interpret_metrics(mape):
    print("\n📊 Model Interpretation:")

    if mape < 10:
        print("Excellent model")
    elif mape < 20:
        print("Good model")
    elif mape < 50:
        print("Average model")
    else:
        print("Poor model")


# =========================================================
# Step 8 — Plot Actual vs Predicted
# =========================================================
def plot_actual_vs_predicted(test, actual, predicted):
    plt.figure(figsize=(15, 6))

    plt.plot(test["ds"], actual, label="Actual")
    plt.plot(test["ds"], predicted, label="Predicted")

    plt.legend()
    plt.title("Actual vs Predicted COVID Cases")
    plt.xlabel("Date")
    plt.ylabel("Cases")

    plt.show()


# =========================================================
# Step 9 — Error Over Time
# =========================================================
def plot_error_over_time(test, actual, predicted):
    errors = actual - predicted

    plt.figure(figsize=(15, 6))

    plt.plot(test["ds"], errors)

    plt.title("Prediction Error Over Time")
    plt.xlabel("Date")
    plt.ylabel("Error")

    plt.axhline(0, color="black", linestyle="--")

    plt.show()

    return errors


# =========================================================
# Step 11 — Residual Distribution
# =========================================================
def plot_error_distribution(actual, predicted):
    errors = actual - predicted

    plt.figure(figsize=(10, 6))

    sns.histplot(errors, bins=50, kde=True)

    plt.title("Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    plt.show()

    return errors


# =========================================================
# Step 12 — Backtesting
# =========================================================
def run_backtesting(model):
    df_cv = cross_validation(
        model,
        initial="200 days",
        period="30 days",
        horizon="30 days"
    )

    df_p = performance_metrics(df_cv)

    print("\n📊 Backtesting Results:")
    print(df_p.head())

    return df_cv, df_p


# =========================================================
# Step 13 — Final Performance Summary
# =========================================================
def print_performance_summary(mae, rmse, mape):
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("----------------------------")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)


# =========================================================
# Step 14 — Save Evaluation Report
# =========================================================
def save_evaluation_report(mae, rmse, mape):
    report = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    pd.DataFrame([report]).to_csv(
        "outputs/evaluation_metrics.csv",
        index=False
    )

    print("\n✅ Evaluation report saved to outputs/evaluation_metrics.csv")