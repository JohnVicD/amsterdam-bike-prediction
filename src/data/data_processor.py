import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def load_data(self, bike_file, weather_file):
        """Load data from CSV files"""
        print("Loading bike and weather data...")
        bike_data = pd.read_csv(bike_file)
        weather_data = pd.read_csv(weather_file)

        # Convert timestamps
        bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'])
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

        return bike_data, weather_data

    def merge_data(self, bike_data, weather_data):
        """
        Merge bike and weather data on timestamp
        """
        # Ensure timestamps are in datetime format
        bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp'])
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

        # Round timestamps to nearest hour for merging
        bike_data['timestamp_hour'] = bike_data['timestamp'].dt.floor('H')
        weather_data['timestamp_hour'] = weather_data['timestamp'].dt.floor('H')

        # Merge data
        merged_data = pd.merge(
            bike_data,
            weather_data,
            left_on='timestamp_hour',
            right_on='timestamp_hour',
            how='inner'
        )

        # Prioritize the original timestamp column from bike data
        merged_data['timestamp'] = merged_data['timestamp_x']

        # Drop unnecessary columns
        columns_to_drop = ['timestamp_x', 'timestamp_y', 'timestamp_hour_x', 'timestamp_hour_y']
        merged_data = merged_data.drop(columns=columns_to_drop, errors='ignore')

        # Print debug information
        print("Merged data columns:", list(merged_data.columns))
        print("Merged data shape:", merged_data.shape)

        return merged_data

    def create_time_features(self, df):
        """Create time-based features"""
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Add weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Add rush hour indicator
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

        return df

    def normalize_features(self, df):
        """Normalize numerical features"""
        features_to_normalize = ['temperature', 'wind_speed', 'count']
        df[features_to_normalize] = self.scaler.fit_transform(df[features_to_normalize])
        return df

    def process_data(self, bike_file, weather_file, save_path=None):
        """
        Complete data processing pipeline
        """
        # Load data
        bike_data, weather_data = self.load_data(bike_file, weather_file)

        # Validate data
        if len(bike_data) == 0 or len(weather_data) == 0:
            raise ValueError("Bike or weather data is empty. Check input files.")

        # Merge data
        merged_data = self.merge_data(bike_data, weather_data)

        # Handle missing values
        merged_data = self.handle_missing_values(merged_data)

        # Create time features
        merged_data = self.create_time_features(merged_data)

        # Normalize features
        merged_data = self.normalize_features(merged_data)

        if save_path:
            merged_data.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}")

        return merged_data

    def handle_missing_values(self, df):
        """Enhanced missing value handling"""
        # First, try to fill NaNs using forward and backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')

        # If any NaNs remain, use mean/median imputation
        weather_fills = {
            'temperature': df['temperature'].median(),
            'humidity': df['humidity'].median() if 'humidity' in df.columns else 0,
            'wind_speed': df['wind_speed'].median(),
            'rain': 0,
            'cloud_cover': df['cloud_cover'].median()
        }

        df = df.fillna(weather_fills)

        # As a final step, drop any remaining rows with NaNs
        df = df.dropna()

        return df

    def save_locations(self, locations, filename):
        """
        Save location data to a CSV file
        """
        if locations is not None:
            locations.to_csv(filename, index=False)
            print(f"Locations data saved to {filename}")
        else:
            print("No location data to save")
