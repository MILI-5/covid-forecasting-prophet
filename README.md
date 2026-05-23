# 📊 Time Series Epidemic Forecasting using Facebook Prophet

# 🦠 COVID-19 Epidemic Forecasting using Facebook Prophet

![Python](https://img.shields.io/badge/Python-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![Pandas](https://img.shields.io/badge/Pandas-black)
![Prophet](https://img.shields.io/badge/Prophet-green)

## 🚀 Overview
This project is an end-to-end **epidemic forecasting system** built using WHO COVID-19 data.  
It predicts **30-day future case trends** using **Facebook Prophet** and provides an **interactive Streamlit dashboard** for visualization and analysis.

The goal is to demonstrate **real-world time-series forecasting, ML pipeline design, and deployment-ready AI systems**.

---

## 🧠 Key Features

- 📊 End-to-end data preprocessing pipeline  
- 📈 Time-series forecasting using Facebook Prophet  
- 🔮 30-day future predictions  
- 🌐 Interactive Streamlit dashboard  
- 📉 Actual vs Forecast visualization  
- 📦 Uncertainty intervals (upper/lower bounds)  
- 📊 Model evaluation (MAE, RMSE, MAPE)  
- 📥 Downloadable forecast results  
- 🧠 Insight generation for epidemic trends  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Facebook Prophet  
- Streamlit  
- Plotly  
- Scikit-learn  
- Joblib  

---

## 📂 Dataset

- WHO COVID-19 Global Dataset  
- Daily confirmed cases data  
- Preprocessed into time-series format (`ds`, `y`)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/MILI-5/covid-forecasting-prophet.git
cd covid-forecasting-prophet

## 📊 Model Workflow

- Data Cleaning & Preprocessing  
- Feature Engineering (lags, rolling stats, growth rate)  
- Train Facebook Prophet model  
- Generate 30-day forecast  
- Visualize predictions + uncertainty  
- Evaluate performance metrics  

## 📈 Results

- 🎯 MAPE: ~6% – 12% (Good forecasting accuracy)  
- 📅 Forecast Horizon: 30 days  
- 📉 Captures epidemic waves and trend shifts  
- 📊 Stable seasonal pattern detection  

## 📷 Dashboard Preview

outputs/screenshots/dashboard.jpeg
outputs/screenshots/KPI metrics and trends.jpeg

Example:
- Trend visualization  
- Forecast graph  
- Actual vs predicted  
- Uncertainty bands  

## 🧠 System Design Diagram


## 🌐 Live Demo

👉 https://covid-forecasting-prophet-qfmvockmcntmthbbhvvcpl.streamlit.app/

## 📁 Project Structure

covid-forecasting-prophet/
│
├── app/
│   └── main.py
├── src/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── outputs/
│   ├── forecasts/
│   ├── plots/
│
├── requirements.txt
└── README.md

## 🧠 Business Impact

- Helps in healthcare resource planning  
- Predicts epidemic wave patterns  
- Assists in early warning systems  
- Supports policy-level decision making  

## 👨‍💻 Author

Built as an internship-level AI/ML project demonstrating:

- Time series forecasting  
- ML pipeline engineering  
- Dashboard deployment  
- Real-world data science workflow  

## ⭐ Future Improvements

- Multi-country forecasting system  
- Real-time API data integration  
- LSTM / Transformer comparison models  
- Cloud deployment (AWS / Azure)  