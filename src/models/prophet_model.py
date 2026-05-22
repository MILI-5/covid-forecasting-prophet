import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# =========================================================
# Step 1 — Load Data
# =========================================================
def load_prophet_data():
    df = pd.read_csv("data/processed/india_covid_clean.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")
    df = df[["ds", "y"]]
    return df


# =========================================================
# Step 2 — Visual Check Before Modeling
# =========================================================
def visualize_data(df):
    plt.figure(figsize=(15, 5))
    plt.plot(df["ds"], df["y"])
    plt.title("COVID-19 Daily Cases (India)")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.show()


# =========================================================
# Step 3–5 — Train Model (FULL DATA)
# =========================================================
def train_prophet_model(df):
    """
    Step 20 — INTERNSHIP BOOST (Changepoint Tuning)
    """

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1   # Step 20 added here
    )

    model.fit(df)
    return model


# =========================================================
# Step 6–8 — Forecast
# =========================================================
def make_forecast(model, periods=30):
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    print("\n📊 Forecast Preview:")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    return forecast


# =========================================================
# Step 9 — Forecast Visualization
# =========================================================
def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title("30-Day COVID Forecast")
    plt.show()

    model.plot_components(forecast)
    plt.show()


# =========================================================
# Step 10 — Actual vs Predicted
# =========================================================
def plot_actual_vs_forecast(df, forecast):
    plt.figure(figsize=(15, 6))

    plt.plot(df["ds"], df["y"], label="Actual")
    plt.plot(forecast["ds"], forecast["yhat"], label="Predicted")

    plt.legend()
    plt.title("Actual vs Forecast")
    plt.show()


# =========================================================
# Step 13 — Train on Train Set Only
# =========================================================
def train_on_train_set(train):
    model = Prophet()
    model.fit(train)
    return model


# =========================================================
# Step 14 — Predict on Test Period
# =========================================================
def evaluate_on_test(model, test):
    future_test = model.make_future_dataframe(
        periods=len(test),
        freq="D"
    )

    forecast_test = model.predict(future_test)

    forecast_test = forecast_test.tail(len(test))

    print("\n📊 Test Forecast Preview:")
    print(forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())

    return forecast_test


# =========================================================
# Step 17 — Reusable Forecast Function
# =========================================================
def forecast_cases(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast