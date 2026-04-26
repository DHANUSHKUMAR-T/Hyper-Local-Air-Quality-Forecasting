import pandas as pd
import joblib
from data_agent import DataFusionAgent
from calibration_agent import SensorCalibrationAgent
from feature_agent import FeatureEngineeringAgent
from spatial_agent import SpatialIntelligenceAgent
from risk_agent import RiskAssessmentAgent

class AirQualityPipeline:
    """
    Main orchestrator for the Hyper-Local Air Quality Forecasting Pipeline.
    """
    
    def __init__(self):
        self.data_agent = DataFusionAgent()
        self.risk_agent = RiskAssessmentAgent()
        
    def refresh_data(self):
        """Run full pipeline from raw simulation to model-ready features."""
        print("[1/5] Simulating Multi-Source Data...")
        df_raw = self.data_agent.generate_full_dataset()
        df_raw.to_csv('fused_aqi_data.csv', index=False)
        
        print("[2/5] Running Sensor Calibration...")
        cal_agent = SensorCalibrationAgent()
        df_cal = cal_agent.calibrate()
        df_cal.to_csv('calibrated_aqi_data.csv', index=False)
        
        print("[3/5] Engineering Advanced Features...")
        feat_agent = FeatureEngineeringAgent()
        df_ready = feat_agent.engineer_features()
        df_ready.to_csv('model_ready_data.csv', index=False)
        
        print("[4/5] Risk Assessment...")
        df_ready = self.risk_agent.process_data(df_ready)
        df_ready.to_csv('model_ready_data.csv', index=False)
        
        print("[5/5] Pipeline Refresh Complete.")
        return df_ready

    def get_forecast_grid(self, target='target_6h'):
        """Generate spatial grid for the most recent forecast."""
        spatial_agent = SpatialIntelligenceAgent()
        lon, lat, z = spatial_agent.interpolate(target=target)
        return lon, lat, z

if __name__ == "__main__":
    pipeline = AirQualityPipeline()
    pipeline.refresh_data()
