import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os


class WeatherDataCollector:
    def __init__(self):
        self.base_url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
        self.station_id = 240  # Schiphol

    def get_weather_data(self, start_date, end_date):
        """Get weather data from KNMI"""
        print("\n=== Starting Weather Collection ===")
        print(f"Requested date range: {start_date} to {end_date}")

        # Get daily data first
        daily_data = self._get_daily_data(start_date, end_date)
        if daily_data is None:
            return None

        print(f"\nConverting {len(daily_data)} days to hourly data...")
        # Convert to hourly - this is where we were missing the conversion!
        hourly_data = self._convert_to_hourly(daily_data, start_date, end_date)

        print(f"\nFinal data shape: {hourly_data.shape}")
        return hourly_data

    def _get_daily_data(self, start_date, end_date):
        """Get daily weather data from KNMI"""
        params = {
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'inseason': 0,
            'vars': 'TG:RH:FG:NG',
            'stns': self.station_id,
        }

        try:
            print(f"Fetching KNMI data...")
            response = requests.post(self.base_url, data=params)
            response.raise_for_status()

            # Process response
            lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            data_lines = [line for line in lines if line.lstrip().startswith(str(self.station_id))]

            if not data_lines:
                raise Exception("No data found in response")

            # Create DataFrame
            df = pd.DataFrame([line.split(',') for line in data_lines])
            df.columns = ['STN', 'YYYYMMDD', 'TG', 'RH', 'FG', 'NG']

            # Process the daily data
            daily_data = self._process_daily_data(df)
            return daily_data

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    def _process_daily_data(self, df):
        """Process daily weather data"""
        print("Processing daily data...")

        # Clean and convert data
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Convert date string to datetime
        df['timestamp'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')

        # Convert units
        df['temperature'] = pd.to_numeric(df['TG']) / 10.0  # 0.1°C to °C
        df['rain'] = pd.to_numeric(df['RH']) / 10.0  # 0.1mm to mm
        df['wind_speed'] = pd.to_numeric(df['FG']) / 10.0  # 0.1 m/s to m/s
        df['cloud_cover'] = pd.to_numeric(df['NG'])

        # Replace -0.1 rain values with 0.05
        df.loc[df['rain'] == -0.1, 'rain'] = 0.05

        return df[['timestamp', 'temperature', 'rain', 'wind_speed', 'cloud_cover']]

    def _convert_to_hourly(self, daily_data, start_date, end_date):
        """Convert daily data to hourly with realistic patterns"""
        # Create hourly timestamp range for the EXACT period needed
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')
        print(f"Creating hourly data from {hourly_range[0]} to {hourly_range[-1]}")

        # Initialize hourly DataFrame
        hourly_data = pd.DataFrame(index=hourly_range)
        hourly_data.index.name = 'timestamp'

        # Interpolate for each day
        for day_data in daily_data.itertuples():
            day_start = day_data.timestamp
            day_end = day_start + pd.Timedelta(days=1)
            day_mask = (hourly_data.index >= day_start) & (hourly_data.index < day_end)

            # Create daily patterns
            hours = hourly_data.index[day_mask].hour

            # Temperature: daily curve
            temp_variation = -2 * np.cos(2 * np.pi * (hours - 14) / 24)
            hourly_data.loc[day_mask, 'temperature'] = day_data.temperature + temp_variation

            # Rain: distribute across random hours
            if day_data.rain > 0:
                rain_hours = np.random.choice(24, size=min(8, max(1, int(day_data.rain))), replace=False)
                for hour in rain_hours:
                    hour_mask = (hourly_data.index >= day_start + pd.Timedelta(hours=hour)) & \
                                (hourly_data.index < day_start + pd.Timedelta(hours=hour + 1))
                    hourly_data.loc[hour_mask, 'rain'] = day_data.rain / len(rain_hours)
                hourly_data['rain'].fillna(0, inplace=True)
            else:
                hourly_data.loc[day_mask, 'rain'] = 0

            # Wind: random variations
            wind_variation = np.random.normal(0, 0.5, size=len(hours))
            hourly_data.loc[day_mask, 'wind_speed'] = day_data.wind_speed + wind_variation

            # Cloud cover: constant through day
            hourly_data.loc[day_mask, 'cloud_cover'] = day_data.cloud_cover

        # Reset index to make timestamp a column
        hourly_data = hourly_data.reset_index()

        print(f"Created {len(hourly_data)} hourly records")
        return hourly_data

    def save_data(self, data, filename):
        """Save weather data to CSV file"""
        if data is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            data.to_csv(filename, index=False)
            print(f"Weather data saved to {filename}")
            print(f"Saved {len(data)} records from {data['timestamp'].min()} to {data['timestamp'].max()}")
        else:
            print("No data to save")