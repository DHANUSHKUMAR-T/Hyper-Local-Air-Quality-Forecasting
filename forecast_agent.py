import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib

class ForecastingAgent:
    """
    Agent responsible for training forecasting models (Baseline vs Advanced)
    and estimating uncertainty.
    """

    def __init__(self, data_path='model_ready_data.csv'):
        self.df = pd.read_csv(data_path)
        self.features = [
            'temperature', 'humidity', 'wind_speed', 'traffic_density', 'industrial_activity',
            'hour', 'day_of_week', 'is_weekend', 'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6',
            'pm25_rolling_3', 'pm25_rolling_6', 'wind_x', 'wind_y',
            'traffic_wind_interaction', 'humidity_pollution_interaction'
        ]
        self.targets = ['target_6h', 'target_12h', 'target_24h']
        self.models = {}

    def train_baseline(self, target):
        """Train Linear Regression baseline."""
        X = self.df[self.features]
        y = self.df[target]
        
        # Simple split (last 20% as test)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)
        
        return lr, rmse, mape

    def train_advanced(self, target):
        """Train XGBoost Advanced model with Quantile Regression for uncertainty."""
        X = self.df[self.features]
        y = self.df[target]
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Standard XGBoost for point prediction
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Quantile XGBoost for 95% confidence interval (lower: 0.025, upper: 0.975)
        # Note: In XGBoost >= 2.0, we use quantile_alpha parameter
        model_low = xgb.XGBRegressor(n_estimators=100, objective='reg:quantileerror', quantile_alpha=0.025)
        model_high = xgb.XGBRegressor(n_estimators=100, objective='reg:quantileerror', quantile_alpha=0.975)
        
        model_low.fit(X_train, y_train)
        model_high.fit(X_train, y_train)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)
        
        return (model, model_low, model_high), rmse, mape

    def run(self):
        """Execute training for all targets."""
        results = []
        for target in self.targets:
            print(f"\n--- Training for {target} ---")
            
            # Baseline
            lr_model, lr_rmse, lr_mape = self.train_baseline(target)
            print(f"LR Baseline | RMSE: {lr_rmse:.2f} | MAPE: {lr_mape*100:.2f}%")
            
            # Advanced
            (xgb_m, xgb_l, xgb_h), xgb_rmse, xgb_mape = self.train_advanced(target)
            print(f"XGB Advanced | RMSE: {xgb_rmse:.2f} | MAPE: {xgb_mape*100:.2f}%")
            
            # Save models
            self.models[target] = {'lr': lr_model, 'xgb': xgb_m, 'xgb_low': xgb_l, 'xgb_high': xgb_h}
            joblib.dump(self.models[target], f'models_{target}.pkl')
            
            results.append({
                'target': target,
                'lr_rmse': lr_rmse, 'lr_mape': lr_mape,
                'xgb_rmse': xgb_rmse, 'xgb_mape': xgb_mape,
                'improvement_mape': (lr_mape - xgb_mape) / lr_mape * 100
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    agent = ForecastingAgent()
    summary = agent.run()
    print("\nTraining Summary:")
    print(summary)
    summary.to_csv('model_evaluation.csv', index=False)
