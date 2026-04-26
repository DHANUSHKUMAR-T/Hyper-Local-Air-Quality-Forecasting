import pandas as pd
import numpy as np

class FeatureEngineeringAgent:
    """
    Agent responsible for creating advanced temporal, spatial, and meteorological features.
    """

    def __init__(self, data_path='calibrated_aqi_data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['station_id', 'timestamp'])

    def engineer_features(self):
        """Create features for each station."""
        featured_dfs = []
        
        for station_id in self.df['station_id'].unique():
            sdf = self.df[self.df['station_id'] == station_id].copy()
            
            # --- Temporal Features ---
            sdf['hour'] = sdf['timestamp'].dt.hour
            sdf['day_of_week'] = sdf['timestamp'].dt.dayofweek
            sdf['is_weekend'] = sdf['day_of_week'].isin([5, 6]).astype(int)
            
            # --- Lag Features (t-1, t-3, t-6) ---
            # Using calibrated PM2.5
            sdf['pm25_lag_1'] = sdf['pm25_calibrated_xgb'].shift(1)
            sdf['pm25_lag_3'] = sdf['pm25_calibrated_xgb'].shift(3)
            sdf['pm25_lag_6'] = sdf['pm25_calibrated_xgb'].shift(6)
            
            # --- Rolling Averages (3h, 6h) ---
            sdf['pm25_rolling_3'] = sdf['pm25_calibrated_xgb'].rolling(window=3).mean()
            sdf['pm25_rolling_6'] = sdf['pm25_calibrated_xgb'].rolling(window=6).mean()
            
            # --- Meteorological Features ---
            # Wind vector components
            rad = np.radians(sdf['wind_dir'])
            sdf['wind_x'] = sdf['wind_speed'] * np.cos(rad)
            sdf['wind_y'] = sdf['wind_speed'] * np.sin(rad)
            
            # Interactions
            sdf['traffic_wind_interaction'] = sdf['traffic_density'] / (sdf['wind_speed'] + 1)
            sdf['humidity_pollution_interaction'] = sdf['humidity'] * sdf['pm25_calibrated_xgb'] / 100
            
            # --- Targets (6h, 12h, 24h forecast) ---
            sdf['target_6h'] = sdf['pm25_calibrated_xgb'].shift(-6)
            sdf['target_12h'] = sdf['pm25_calibrated_xgb'].shift(-12)
            sdf['target_24h'] = sdf['pm25_calibrated_xgb'].shift(-24)
            
            featured_dfs.append(sdf)
            
        final_df = pd.concat(featured_dfs, ignore_index=True)
        # Drop rows with NaN (due to lags and targets)
        final_df = final_df.dropna()
        return final_df

if __name__ == "__main__":
    agent = FeatureEngineeringAgent()
    featured_df = agent.engineer_features()
    featured_df.to_csv('model_ready_data.csv', index=False)
    print(f"Feature engineering complete. Dataset shape: {featured_df.shape}")
    print(featured_df.head())
