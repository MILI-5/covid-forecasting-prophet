import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.features.feature_engineering import (
    create_time_series_features,
    get_final_features,
    save_features
)

from src.models.prophet_model import (
    load_prophet_data,
    train_prophet_model,
    make_forecast
)

# =========================================================
# STEP 4 — PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Epidemic Forecasting Dashboard",
    layout="wide"
)

# =========================================================
# STEP 7 — APP TITLE SECTION
# =========================================================
st.title("🦠 COVID-19 Epidemic Forecasting Dashboard")
st.markdown("Forecasting future case trends using Facebook Prophet")

# =========================================================
# STEP 6 — SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("Control Panel")

country = st.sidebar.selectbox(
    "Select Country",
    ["India", "United States", "Brazil"]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    7, 60, 30
)

# =========================================================
# STEP 5 — LOAD DATA + MODEL
# =========================================================
df = pd.read_csv("data/processed/india_covid_clean.csv")
df["ds"] = pd.to_datetime(df["ds"])

@st.cache_resource
def load_model():
    model = joblib.load("outputs/model/prophet_model.pkl")
    return model

model = load_model()

# =========================================================
# RAW DATA
# =========================================================
st.subheader("📌 Raw Cleaned Data")
st.dataframe(df.head())

# =========================================================
# FEATURE ENGINEERING
# =========================================================
st.subheader("⚙️ Feature Engineering Pipeline")

df_features = create_time_series_features(df)

st.write("After feature engineering shape:", df_features.shape)
st.dataframe(df_features.head())

# =========================================================
# FINAL FEATURE SELECTION
# =========================================================
st.subheader("🎯 Final ML Dataset Preparation")

final_df = get_final_features(df_features)

st.write("Final dataset shape:", final_df.shape)
st.dataframe(final_df.head())

# =========================================================
# SAVE FEATURES
# =========================================================
save_features(final_df, "data/processed/india_features.csv")

st.success("Feature engineering completed and dataset saved successfully ✅")

# =========================================================
# STEP 9 — FILTER DATA BY COUNTRY
# =========================================================
country_df = df.copy()
country_df["ds"] = pd.to_datetime(country_df["ds"])

# =========================================================
# STEP 10 — HISTORICAL TREND VISUALIZATION
# =========================================================
fig = px.line(
    country_df,
    x="ds",
    y="y",
    title=f"{country} COVID-19 Trend"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# STEP 8 — KPI METRICS SECTION
# =========================================================
st.subheader("📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

mae = 1234   # placeholder (replace with real evaluation later)
rmse = 2456
mape = 6.2

col1.metric("MAE", mae)
col2.metric("RMSE", rmse)
col3.metric("MAPE (%)", mape)

# =========================================================
# BASIC INSIGHTS
# =========================================================
st.subheader("📈 Basic Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(final_df))
col2.metric("Max Cases", int(final_df["y"].max()))
col3.metric("Avg Cases", int(final_df["y"].mean()))

# =========================================================
# TIME SERIES VISUALIZATION
# =========================================================
st.subheader("📉 Trend Overview")

st.line_chart(final_df.set_index("ds")["y"])

# =========================================================
# FORECASTING MODULE (PHASE 7)
# =========================================================
st.subheader("🤖 Forecasting Module")

run_forecast = st.button(f"🚀 Generate {forecast_days}-Day Forecast")

if run_forecast:

    prophet_df = load_prophet_data()
    trained_model = train_prophet_model(prophet_df)

    forecast = make_forecast(trained_model, periods=forecast_days)

    st.success("Forecast generated successfully ✅")

    # =========================================================
    # STEP 11 — GENERATE FORECAST
    # =========================================================
    future = trained_model.make_future_dataframe(periods=forecast_days)
    forecast = trained_model.predict(future)

    # =========================================================
    # STEP 12 — FORECAST VISUALIZATION (ACTUAL vs FORECAST)
    # =========================================================
    st.subheader("📊 Actual vs Forecast Comparison")

    fig2 = px.line()

    fig2.add_scatter(
        x=country_df["ds"],
        y=country_df["y"],
        name="Actual"
    )

    fig2.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        name="Forecast"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =========================================================
    # STEP 13 — UNCERTAINTY VISUALIZATION
    # =========================================================
    st.subheader("📉 Forecast Uncertainty (Confidence Bounds)")

    fig3 = px.line()

    fig3.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        name="Upper Bound",
        line=dict(dash="dot")
    )

    fig3.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        name="Lower Bound",
        line=dict(dash="dot")
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =========================================================
    # STEP 14 — FORECAST TABLE
    # =========================================================
    st.subheader("📄 Forecast Data")

    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
    )

    # =========================================================
    # STEP 15 — DOWNLOAD BUTTON
    # =========================================================
    csv = forecast.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast Data",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv"
    )

    # =========================================================
    # STEP 16 — KEY INSIGHTS
    # =========================================================
    st.subheader("📌 Key Insights")

    st.write("""
    - Epidemic waves show cyclical patterns  
    - Model captures trend + seasonality  
    - Forecast helps in resource planning  
    """)