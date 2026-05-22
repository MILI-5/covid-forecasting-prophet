import streamlit as st
import pandas as pd

from src.features.feature_engineering import (
    create_time_series_features,
    get_final_features,
    save_features
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="COVID Forecasting Dashboard", layout="wide")

st.title("📊 COVID-19 Forecasting Dashboard")
st.write("Internship-level Time Series Forecasting Project (Full Feature Pipeline)")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/india_covid_clean.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = load_data()

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
# SAVE DATASET
# =========================================================
save_features(final_df, "data/processed/india_features.csv")

st.success("Feature engineering completed and dataset saved successfully ✅")

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