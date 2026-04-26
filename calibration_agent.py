import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance

class SensorCalibrationAgent:
    """
    Agent responsible for correcting low-cost sensor drift using CAAQMS reference.
    """

    def __init__(self, data_path='fused_aqi_data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.models = {}
        self.reference_data = self.df[self.df['station_type'] == 'reference']
        self.lcs_data = self.df[self.df['station_type'] == 'low_cost']

    def find_nearest_reference(self, lcs_lat, lcs_lon):
        """Find the nearest reference station for an LCS."""
        ref_coords = self.reference_data[['lat', 'lon']].drop_duplicates()
        distances = ref_coords.apply(lambda row: distance.euclidean((lcs_lat, lcs_lon), (row['lat'], row['lon'])), axis=1)
        nearest_idx = distances.idxmin()
        # Find the station_id for this coordinate
        ref_station = self.reference_data[(self.reference_data['lat'] == ref_coords.loc[nearest_idx, 'lat']) & 
                                          (self.reference_data['lon'] == ref_coords.loc[nearest_idx, 'lon'])]['station_id'].iloc[0]
        return ref_station

    def calibrate(self):
        """Train and apply calibration models."""
        calibrated_dfs = []
        
        # Reference mapping
        lcs_stations = self.lcs_data['station_id'].unique()
        
        for lcs_id in lcs_stations:
            lcs_df = self.lcs_data[self.lcs_data['station_id'] == lcs_id].copy()
            ref_id = self.find_nearest_reference(lcs_df['lat'].iloc[0], lcs_df['lon'].iloc[0])
            ref_df = self.reference_data[self.reference_data['station_id'] == ref_id].copy()
            
            # Align by timestamp
            aligned = pd.merge(lcs_df, ref_df[['timestamp', 'pm25']], on='timestamp', suffixes=('_lcs', '_ref'))
            
            # Training features for calibration
            X = aligned[['pm25_lcs', 'temperature', 'humidity']]
            y = aligned['pm25_ref']
            
            # 1. Baseline: Linear Regression
            lr = LinearRegression()
            lr.fit(X, y)
            aligned['pm25_calibrated_lr'] = lr.predict(X)
            
            # 2. Advanced: XGBoost
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
            xgb.fit(X, y)
            aligned['pm25_calibrated_xgb'] = xgb.predict(X)
            
            # Metrics
            rmse_raw = np.sqrt(mean_squared_error(aligned['pm25_ref'], aligned['pm25_lcs']))
            rmse_lr = np.sqrt(mean_squared_error(aligned['pm25_ref'], aligned['pm25_calibrated_lr']))
            rmse_xgb = np.sqrt(mean_squared_error(aligned['pm25_ref'], aligned['pm25_calibrated_xgb']))
            
            print(f"Station {lcs_id} | Raw RMSE: {rmse_raw:.2f} | LR RMSE: {rmse_lr:.2f} | XGB RMSE: {rmse_xgb:.2f}")
            
            calibrated_dfs.append(aligned)
            
        final_cal_df = pd.concat(calibrated_dfs, ignore_index=True)
        return final_cal_df

if __name__ == "__main__":
    agent = SensorCalibrationAgent()
    calibrated_df = agent.calibrate()
    # Save the calibrated data
    calibrated_df.to_csv('calibrated_aqi_data.csv', index=False)
    print("\nCalibration complete. Saved to calibrated_aqi_data.csv")
