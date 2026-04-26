import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import joblib
from pipeline import AirQualityPipeline
from risk_agent import RiskAssessmentAgent
from datetime import datetime

st.set_page_config(page_title="Hyper-Local AQI Forecaster", layout="wide")

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return AirQualityPipeline()

pipeline = get_pipeline()
risk_agent = RiskAssessmentAgent()

# Title and Description
st.title("🌍 Tamil Nadu Air Quality Intelligence")
st.markdown("### Hyper-Local PM2.5 Forecasting & Health Alerts - Chennai-Ennore Corridor")

# Data loading
@st.cache_data
def load_data():
    return pd.read_csv('model_ready_data.csv')

df = load_data()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Refresh Data Pipeline"):
    with st.spinner("Re-training and updating..."):
        pipeline.refresh_data()
        st.rerun()

target_horizon = st.sidebar.selectbox("Forecast Horizon", ["target_6h", "target_12h", "target_24h"])
selected_station = st.sidebar.selectbox("Select Station", df['station_id'].unique())

# --- Dashboard Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Street-Level AQI Heatmap ({target_horizon})")
    lon_grid, lat_grid, z_grid = pipeline.get_forecast_grid(target=target_horizon)
    
    # Create Folium Map
    m = folium.Map(location=[13.15, 80.25], zoom_start=11, tiles="CartoDB positron")
    
    # Add Heatmap Overlay using the grid
    # For a simple prototype, we'll plot the grid points as circles with opacity
    # because true contour overlays in folium are more complex.
    # We sample every 2nd point for performance
    step = 2
    for i in range(0, z_grid.shape[0], step):
        for j in range(0, z_grid.shape[1], step):
            val = z_grid[i, j]
            aqi = risk_agent.calculate_aqi(val)
            risk = risk_agent.get_risk_level(aqi)
            color = "#00e400" if risk == "Good" else "#ffff00" if risk == "Satisfactory" else "#ff7e00" if risk == "Moderate" else "#ff0000" if risk == "Poor" else "#8f3f97" # "Very Poor" simplified
            
            folium.CircleMarker(
                location=[lat_grid[i, j], lon_grid[i, j]],
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.4,
                weight=0,
                popup=f"AQI: {aqi:.1f} ({risk})"
            ).add_to(m)
            
    st_folium(m, width=900, height=500)

with col2:
    st.subheader("Personalized Risk Alerts")
    station_data = df[df['station_id'] == selected_station].iloc[-1]
    pm25_val = station_data[target_horizon]
    aqi_val = risk_agent.calculate_aqi(pm25_val)
    risk_level = risk_agent.get_risk_level(aqi_val)
    
    # Display Alert Cards
    st.metric(label="Predicted AQI", value=f"{aqi_val:.0f}", delta=risk_level)
    
    alerts = risk_agent.generate_alerts(pm25_val, aqi_val, risk_level)
    
    for persona, msg in alerts.items():
        with st.expander(f"🔔 {persona} Persona", expanded=True):
            st.write(msg)

# --- Forecasting Trends ---
st.subheader(f"24-Hour Trends for {selected_station}")
hist_df = df[df['station_id'] == selected_station].tail(24)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['pm25_calibrated_xgb'], name="Calibrated PM2.5", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['target_6h'], name="6h Forecast", line=dict(dash='dash', color='orange')))
fig.update_layout(xaxis_title="Time", yaxis_title="PM2.5 (µg/m³)")
st.plotly_chart(fig, use_container_width=True)

# --- Performance vs Baseline ---
st.subheader("Model Performance (Advanced vs Baseline)")
metrics_df = pd.read_csv('model_evaluation.csv')
st.table(metrics_df)

st.info("💡 **Advanced Model (XGBoost)** leverages multi-source features (Industrial, Traffic, Weather) to outperform the **Baseline (Linear Regression)**.")
