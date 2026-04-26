# 🌍 Hyper-Local Air Quality Forecasting  
### 🚀 TNSDC Naan Mudhalvan 2026 – Level 2 Prototype

---

## 📌 Overview
This project builds a **data-centric AI system** to predict **street-level air quality (PM2.5 / AQI)** 6–24 hours ahead in Tamil Nadu’s industrial corridors (e.g., Manali–Ennore, Tirupur, Thoothukudi).

Unlike traditional city-level AQI, this system provides **hyper-local predictions and personalized health alerts** by fusing multiple real-world data sources.

---

## 🎯 Problem Statement
Predict street-level air quality using sparse TNPCB monitoring stations by integrating weather, traffic, and industrial signals to generate actionable exposure alerts.

---

## 🧠 Key Features
- Multi-source data fusion (CAAQMS + Weather + Traffic + Industry)
- Sensor calibration (corrects low-cost sensor drift)
- Time-series forecasting (6–24 hour prediction)
- Spatial interpolation (street-level AQI mapping)
- Personalized risk alerts (asthmatics, children, general public)
- Baseline vs improved model comparison
- Interactive dashboard (Streamlit)

---

## 🏗️ System Architecture

Raw Data Sources
(CAAQMS + Sensors + Weather + Traffic + Industry)
↓
Data Preprocessing & Cleaning
↓
Sensor Calibration Model
↓
Feature Engineering
↓
Baseline Model (Linear Regression)
↓
Advanced Model (XGBoost / LSTM)
↓
Spatial Interpolation (IDW / Kriging)
↓
Risk Scoring Module
↓
Streamlit Dashboard


---

## 📊 Dataset Description

| Source | Description |
|------|------------|
| CAAQMS | Hourly air quality data (PM2.5, PM10, NO2, etc.) |
| Low-cost Sensors | 15-min interval data with drift |
| Weather (IMD) | Wind, temperature, humidity |
| Traffic | Road congestion levels |
| Industry | Power plant activity |
| Spatial Data | Land use, roads, industrial zones |

---

## ⚙️ Tech Stack

### Core
- Python 3.10+
- Pandas, NumPy

### Machine Learning
- Scikit-learn
- XGBoost / LightGBM
- TensorFlow (optional LSTM)

### Geospatial
- GeoPandas
- Shapely
- Folium

### Visualization
- Streamlit

<img width="1832" height="911" alt="Image" src="https://github.com/user-attachments/assets/846f187a-59a5-4581-9865-1e5bea91657d" />
<img width="1454" height="346" alt="Image" src="https://github.com/user-attachments/assets/0989c486-4c41-4578-b690-157205f49bc9" />
<img width="1412" height="619" alt="Image" src="https://github.com/user-attachments/assets/be036545-210c-4b7c-8ac3-0b20d1ea49cd" />
