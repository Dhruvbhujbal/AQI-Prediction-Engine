<div align="center">

# 🌍 AQI Prediction Engine

**Meteorological Air Quality Forecasting — powered by Machine Learning & Streamlit**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![WeatherAPI](https://img.shields.io/badge/WeatherAPI-Live_Data-00BCD4?style=for-the-badge&logo=cloud&logoColor=white)](https://www.weatherapi.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**AQI Prediction Engine** is a research-grade, full-stack machine learning application that predicts **PM2.5 concentrations** across Indian cities using *strictly meteorological features* — no pollutant data, no target leakage.

The app is built on a **Random Forest Regressor** trained on real atmospheric data and deployed as a multi-page **Streamlit** dashboard with three distinct prediction modes, live API integration, interactive visualizations, and a full model methodology section.

> **Key design principle:** The model deliberately excludes co-pollutant features (PM10, NO₂, O₃) and the EPA AQI index — which are mathematically derived from PM2.5 — to build a genuine meteorological forecasting tool rather than an identity function.

---

## ✨ Features

- 🧠 **3 prediction modes** — historical monthly forecasting, manual scenario simulation, and live real-time prediction
- 📡 **Live WeatherAPI integration** — fetches real-time weather data and compares model predictions against actual IoT sensor readings
- 📊 **Interactive data dashboard** — top polluted cities, wind dispersion scatter plots, and Pearson correlation heatmaps via Plotly
- 🔬 **Model Methodology page** — explains target leakage fix, R² / RMSE / MAE metrics, and Random Forest feature importances (XAI)
- 🌐 **Streamlit multi-page app** — sidebar navigation across 6 sections
- 🧹 **Zero leakage architecture** — trained only on exogenous weather features

---

## 🗂️ Project Structure

```
AQI-Prediction-Engine/
│
├── .streamlit/                        # Streamlit theme config
├── data/
│   └── aqi_training_data.csv          # Training dataset (weather + PM2.5)
│
├── models/
│   └── best_aqi_model.pkl             # Trained Random Forest model
│
├── app.py                             # Streamlit app — all 6 pages
├── train_model.py                     # Model training & evaluation script
├── requirements.txt
├── Dhruv AQI Doc.docx                 # Project documentation
└── Combine AQI Doc.docx               # Combined project report
```

---

## 🧠 How It Works

### Training Pipeline (`train_model.py`)

| Step | Description |
|------|-------------|
| **Load** | Read `aqi_training_data.csv`, normalize column names |
| **Feature Selection** | 6 strictly meteorological features (no pollutants) |
| **Split** | 80/20 train-test split, `random_state=42` |
| **Train** | `RandomForestRegressor` — 100 estimators |
| **Evaluate** | R², RMSE (µg/m³), MAE (µg/m³) |
| **Export** | Saved as `models/best_aqi_model.pkl` via `joblib` |

### Input Features

| Feature | Unit | Description |
|---------|------|-------------|
| `temperature_celsius` | °C | Ambient air temperature |
| `wind_kph` | kph | Wind speed (key dispersion factor) |
| `humidity` | % | Relative humidity |
| `pressure_mb` | mb | Atmospheric pressure |
| `precip_mm` | mm | Precipitation |
| `visibility_km` | km | Atmospheric visibility |

**Target:** `air_quality_PM2_5` (µg/m³) — raw PM2.5 concentration

---

## 📈 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.5236 | 52.36% of PM2.5 variance explained by weather alone |
| **RMSE** | 65.98 µg/m³ | Root Mean Squared Error |
| **MAE** | 40.28 µg/m³ | Mean Absolute Error |

> The unexplained ~48% variance is attributed to anthropogenic factors absent from the dataset — traffic density, industrial emissions, crop burning, and construction. This is an expected and honest result for a purely meteorological model.

---

## 🖥️ App Pages

### 🏠 Home (Abstract)
Dataset overview with live record count, global averages for PM2.5, PM10, and temperature.

### 📅 Mode 1 — Monthly Forecast
Select a city and month → the app aggregates historical weather averages for that combination and feeds them to the model, returning a predicted average PM2.5 with full explainability metrics.

### 🎛️ Mode 2 — Scenario Simulation
Manual sliders for all 6 weather parameters. Advanced settings (pressure, precipitation, visibility) are collapsed in an expander to keep the UI clean. Returns a styled prediction card with color-coded hazard status.

```
≤ 50 µg/m³  →  ✅ Optimal atmospheric dispersion
≤ 100 µg/m³ →  ⚠️  Moderate accumulation detected
> 100 µg/m³ →  🚨 Critical pollution accumulation hazard
```

### 📡 Mode 3 — Live API Prediction
Enter any city name → live weather data is fetched from WeatherAPI → fed into the trained model → ML prediction vs. actual real-world IoT sensor PM2.5 displayed side-by-side with delta comparison.

### 📊 Data Dashboard
Three interactive Plotly charts:
- Top 10 most polluted cities by average PM2.5 (bar chart)
- Wind speed vs. PM2.5 scatter plot (demonstrating dispersion effect)
- Full Pearson correlation heatmap across all features

### 🧠 Model Methodology
Documents the target leakage problem & fix, evaluation metrics, and Random Forest feature importance weights (Explainable AI) rendered as an interactive horizontal bar chart.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A free [WeatherAPI](https://www.weatherapi.com) key (for Mode 3)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dhruvbhujbal/AQI-Prediction-Engine.git
cd AQI-Prediction-Engine

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

Expected output:
```
🚀 Starting Model Training Process...
📊 Training on XXXX rows, Testing on XXXX rows.
⚙️  Training Random Forest Model...

📈 --- REALISTIC MODEL EVALUATION ---
R-Squared (Accuracy): 0.5236
RMSE (Error Margin): 65.98 µg/m³
MAE (Avg Error): 40.28 µg/m³
------------------------------------

✅ Model successfully saved to 'models/best_aqi_model.pkl'!
```

### Run the App

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**

> For Mode 3 (Live API), paste your WeatherAPI key into `API_KEY` in `app.py` before running.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | scikit-learn — Random Forest Regressor |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Plotly Express |
| **Frontend / App** | Streamlit |
| **Live Data** | WeatherAPI (REST) |
| **Model Persistence** | joblib |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a PR or raise an issue.

---

## 👤 Author

**Dhruv Bhujbal**  
M.Sc. Data Science | Savitribai Phule Pune University  
[![GitHub](https://img.shields.io/badge/GitHub-Dhruvbhujbal-181717?style=flat-square&logo=github)](https://github.com/Dhruvbhujbal)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-dhruvbhujbal2601-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/dhruvbhujbal2601)

---

<div align="center">

*If this project helped you, consider leaving a ⭐ on the repo!*

</div>
