import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import joblib

class SpatialIntelligenceAgent:
    """
    Agent responsible for generating street-level predictions for unseen locations
    using Kriging interpolation.
    """

    def __init__(self, data_path='model_ready_data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # Chennai-Ennore bounding box
        self.lat_min, self.lat_max = 13.0, 13.3
        self.lon_min, self.lon_max = 80.15, 80.35

    def generate_grid(self, resolution=50):
        """Create a lat/lon grid for interpolation."""
        lats = np.linspace(self.lat_min, self.lat_max, resolution)
        lons = np.linspace(self.lon_min, self.lon_max, resolution)
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        return grid_lon, grid_lat

    def interpolate(self, target='target_6h', timestamp=None):
        """
        Interpolate forecast values across the grid for a specific timestamp.
        If timestamp is None, use the most recent one.
        """
        if timestamp is None:
            timestamp = self.df['timestamp'].max()
        else:
            timestamp = pd.to_datetime(timestamp)
            
        snapshot = self.df[self.df['timestamp'] == timestamp].copy()
        
        if snapshot.empty:
            print(f"No data for timestamp {timestamp}")
            return None, None, None
        
        # Coordinates and values for Kriging
        x = snapshot['lon'].values
        y = snapshot['lat'].values
        z = snapshot[target].values
        
        grid_lon, grid_lat = self.generate_grid()
        
        # Ordinary Kriging
        ok = OrdinaryKriging(
            x, y, z, 
            variogram_model='linear',
            verbose=False, 
            enable_plotting=False
        )
        
        z_grid, ss_grid = ok.execute('grid', np.unique(grid_lon), np.unique(grid_lat))
        
        return grid_lon, grid_lat, z_grid

if __name__ == "__main__":
    agent = SpatialIntelligenceAgent()
    glon, glat, gz = agent.interpolate()
    if gz is not None:
        print(f"Spatial interpolation complete. Grid shape: {gz.shape}")
        # Save a sample grid for testing
        np.save('sample_grid_z.npy', np.asarray(gz))
        np.save('sample_grid_lon.npy', np.asarray(glon))
        np.save('sample_grid_lat.npy', np.asarray(glat))
