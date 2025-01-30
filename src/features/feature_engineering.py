import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_time_lags(self, df, target_col='count', lags=(1, 2, 3, 24)):
        """
        Create lag features for the target variable
        """
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df

    def create_rolling_features(self, df, target_col='count', windows=(3, 6, 12, 24)):
        """
        Create rolling mean and std features
        """
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df

    def create_holiday_features(self, df, date_col='timestamp'):
        """
        Create holiday and special event features
        """
        df = df.copy()

        # Convert timestamp to datetime if it's not already
        df[date_col] = pd.to_datetime(df[date_col])

        # Dutch holidays (2024-2025 example)
        holidays = {
            '2024-12-25': 'Christmas',
            '2024-12-26': 'Boxing Day',
            '2025-01-01': 'New Year',
            '2025-04-21': 'Easter Monday',
            '2025-04-27': 'Kings Day',
            # Add more holidays as needed
        }

        # Create holiday flag
        df['is_holiday'] = df[date_col].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)

        # Create season features
        df['season'] = df[date_col].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })

        # One-hot encode season
        df = pd.get_dummies(df, columns=['season'], prefix='season')

        return df

    def create_weather_interaction_features(self, df):
        """
        Create interaction features between weather variables
        """
        df = df.copy()

        # Temperature and rain interaction
        df['temp_rain_interaction'] = df['temperature'] * df['rain']

        # Temperature and wind interaction
        df['temp_wind_interaction'] = df['temperature'] * df['wind_speed']

        # Create weather severity index
        df['weather_severity'] = (
                df['rain'].clip(upper=1) * 3 +  # Rain has highest impact
                (df['wind_speed'] / df['wind_speed'].max()) * 2 +  # Wind has medium impact
                (abs(df['temperature'] - 20) / 20)  # Temperature deviation from ideal (20°C)
        )

        return df

    def create_location_features(self, df, location_col='location_id'):
        """
        Create location-based features
        """
        if location_col in df.columns:
            df = df.copy()

            # Calculate location-specific statistics
            location_stats = df.groupby(location_col)['count'].agg({
                'location_mean': 'mean',
                'location_std': 'std',
                'location_max': 'max'
            }).reset_index()

            # Merge back with original dataframe
            df = df.merge(location_stats, on=location_col)

            # Create normalized count per location
            df['count_vs_location_max'] = df['count'] / df['location_max']

        return df

    def engineer_features(self, df, include_lags=True):
        """
        Run complete feature engineering pipeline
        """
        print("Starting feature engineering pipeline...")

        # Create time-based features
        if include_lags:
            print("Creating time lag features...")
            df = self.create_time_lags(df)
            df = self.create_rolling_features(df)

        print("Creating holiday features...")
        df = self.create_holiday_features(df)

        print("Creating weather interaction features...")
        df = self.create_weather_interaction_features(df)

        print("Creating location features...")
        df = self.create_location_features(df)

        # Remove rows with NaN values created by lag features
        if include_lags:
            df = df.dropna()

        print("Feature engineering completed!")
        return df


if __name__ == "__main__":
    # Example usage
    try:
        # Load processed data
        df = pd.read_csv('data/processed/processed_data.csv')

        # Initialize feature engineer
        feature_engineer = FeatureEngineer()

        # Engineer features
        df_featured = feature_engineer.engineer_features(df)

        # Save engineered features
        df_featured.to_csv('data/processed/featured_data.csv', index=False)
        print("Engineered features saved successfully!")

    except Exception as e:
        print(f"Error in feature engineering process: {e}")