import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataFusionAgent:
    """
    Agent responsible for simulating and fusing multi-source environmental data.
    Simulates Chennai-Ennore corridor characteristic.
    """

    def __init__(self, start_date='2024-01-01', days=30):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.days = days
        self.hours = days * 24
        self.timestamps = [self.start_date + timedelta(hours=i) for i in range(self.hours)]
        
        # Spatial setup (Chennai-Ennore area approx)
        # Lat: 13.0 to 13.3, Lon: 80.2 to 80.4
        self.stations = {
            'CAAQMS_01': {'lat': 13.0827, 'lon': 80.2707, 'type': 'reference'}, # Chennai City center
            'CAAQMS_02': {'lat': 13.2161, 'lon': 80.3235, 'type': 'reference'}, # Ennore Port area
            'CAAQMS_03': {'lat': 13.1500, 'lon': 80.2000, 'type': 'reference'}, # Industrial cluster
        }
        
        # Add 15 LCS stations randomly
        for i in range(1, 16):
            self.stations[f'LCS_{i:02d}'] = {
                'lat': random.uniform(13.0, 13.3),
                'lon': random.uniform(80.15, 80.35),
                'type': 'low_cost'
            }

    def simulate_weather(self):
        """Simulate realistic weather patterns for Tamil Nadu."""
        np.random.seed(42)
        
        # Diurnal temperature (high during day, low at night)
        hour_of_day = np.array([ts.hour for ts in self.timestamps])
        temp = 28 + 5 * np.sin((hour_of_day - 8) * (2 * np.pi / 24)) + np.random.normal(0, 1, self.hours)
        
        # Humidity (inverse to temperature)
        humidity = 70 - 15 * np.sin((hour_of_day - 8) * (2 * np.pi / 24)) + np.random.normal(0, 5, self.hours)
        humidity = np.clip(humidity, 30, 95)
        
        # Wind (Sea breeze effect approx 2-4 PM)
        wind_speed = 5 + 3 * np.sin((hour_of_day - 12) * (2 * np.pi / 24)) + np.random.normal(0, 1, self.hours)
        wind_speed = np.clip(wind_speed, 1, 15)
        
        # Wind direction (predominantly Easterly/North-Easterly)
        wind_dir = (90 + np.random.normal(0, 45, self.hours)) % 360
        
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'wind_dir': wind_dir
        })

    def simulate_traffic_industrial(self):
        """Simulate traffic and industrial cycles."""
        hour_of_day = np.array([ts.hour for ts in self.timestamps])
        
        # Traffic peak hours (8-10 AM, 6-8 PM)
        traffic = 10 + 40 * (np.exp(-(hour_of_day - 9)**2 / 4) + np.exp(-(hour_of_day - 19)**2 / 4))
        traffic += np.random.normal(0, 5, self.hours)
        traffic = np.clip(traffic, 5, 100)
        
        # Industrial (steady with slight night increase/decrease depending on shifts)
        industrial = 50 + 10 * np.sin(hour_of_day * (2 * np.pi / 24)) + np.random.normal(0, 2, self.hours)
        
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'traffic_density': traffic,
            'industrial_activity': industrial
        })

    def generate_full_dataset(self):
        """Generate fused dataset for all stations."""
        weather_df = self.simulate_weather()
        activity_df = self.simulate_traffic_industrial()
        
        all_data = []
        
        for station_id, meta in self.stations.items():
            df = weather_df.copy()
            df['station_id'] = station_id
            df['lat'] = meta['lat']
            df['lon'] = meta['lon']
            df['station_type'] = meta['type']
            
            # Merge activity
            df = df.merge(activity_df, on='timestamp')
            
            # Simulate PM2.5 (Ground Truth)
            # Base + Traffic effect + Industrial effect + Weather effect
            # Higher humidity -> usually higher PM2.5 (haze)
            # Higher wind -> dispersion (lower PM2.5)
            pm25_gt = (20 
                       + 0.5 * df['traffic_density'] 
                       + 0.4 * df['industrial_activity'] 
                       - 1.2 * df['wind_speed'] 
                       + 0.1 * df['humidity']
                       + np.random.normal(0, 5, self.hours))
            
            pm25_gt = np.clip(pm25_gt, 5, 300)
            
            if meta['type'] == 'reference':
                df['pm25'] = pm25_gt # Accurate reading
            else:
                # Add drift and bias to LCS
                bias = random.uniform(5, 15)
                drift = np.linspace(0, random.uniform(10, 20), self.hours)
                temp_factor = 0.5 * (df['temperature'] - 28)
                df['pm25'] = pm25_gt + bias + drift + temp_factor + np.random.normal(0, 8, self.hours)
            
            all_data.append(df)
            
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

if __name__ == "__main__":
    agent = DataFusionAgent()
    data = agent.generate_full_dataset()
    data.to_csv('fused_aqi_data.csv', index=False)
    print(f"Generated fused dataset with {len(data)} rows.")
    print(data.head())
