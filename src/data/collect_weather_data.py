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
        # Get daily data first
        daily_data = self._get_daily_data(start_date, end_date)
        if daily_data is None:
            return None

        # Convert to hourly
        hourly_data = self._convert_to_hourly(daily_data, start_date, end_date)
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
            print("Fetching daily weather data...")
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
        print(f"\nConverting {len(daily_data)} days to hourly data...")

        # Create hourly range for exact period
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')
        hourly_data = pd.DataFrame(index=hourly_range)
        hourly_data.index.name = 'timestamp'
        print(f"Created hourly range from {hourly_range[0]} to {hourly_range[-1]}")
        print(f"Total hours: {len(hourly_range)}")

        # Process each day
        for day_data in daily_data.itertuples():
            # Get exact hours for this day within the requested range
            day_start = max(day_data.timestamp, start_date)
            day_end = min(day_data.timestamp + pd.Timedelta(days=1), end_date)
            day_mask = (hourly_data.index >= day_start) & (hourly_data.index < day_end)

            # Get actual hours in this day's mask
            mask_hours = hourly_data.index[day_mask]
            n_hours = len(mask_hours)

            if n_hours > 0:  # Only process if we have hours in this day
                # Temperature pattern
                hour_of_day = mask_hours.hour
                temp_variation = -2 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
                hourly_data.loc[day_mask, 'temperature'] = day_data.temperature + temp_variation

                # Rain pattern
                if day_data.rain > 0:
                    # Scale number of rain hours based on available hours
                    n_rain_hours = min(8, max(1, int(n_hours * (8 / 24))))
                    rain_hours = np.random.choice(n_hours, size=n_rain_hours, replace=False)
                    rain_amount = day_data.rain / n_rain_hours

                    for hour_idx in rain_hours:
                        hour_start = mask_hours[hour_idx]
                        hour_end = hour_start + pd.Timedelta(hours=1)
                        hour_mask = (hourly_data.index >= hour_start) & (hourly_data.index < hour_end)
                        hourly_data.loc[hour_mask, 'rain'] = rain_amount

                    hourly_data['rain'].fillna(0, inplace=True)
                else:
                    hourly_data.loc[day_mask, 'rain'] = 0

                # Wind pattern
                wind_variation = np.random.normal(0, 0.5, size=n_hours)
                hourly_data.loc[day_mask, 'wind_speed'] = day_data.wind_speed + wind_variation

                # Cloud cover (constant through day)
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
            print(f"\nData saved to {filename}")
            print(f"Records: {len(data)}")
            print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        else:
            print("No data to save")

    def _create_hourly_data(self, daily_data, start_date, end_date):
        """Create hourly data from daily data"""
        print("\nCreating hourly data...")

        # Round start and end dates to hours
        start_date = pd.Timestamp(start_date).floor('H')
        end_date = pd.Timestamp(end_date).floor('H')

        # Create hourly timestamp range
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
        hourly_data = pd.DataFrame(index=hourly_range)
        print(f"Created {len(hourly_data)} hour slots")

        # For each day in daily_data
        for _, day in daily_data.iterrows():
            # Get the day's start and end times
            day_start = day['timestamp']
            day_end = day_start + pd.Timedelta(days=1)

            # Create mask for this day's hours
            mask = (hourly_data.index >= day_start) & (hourly_data.index < day_end)
            day_hours = hourly_data.index[mask].hour

            if len(day_hours) > 0:  # Only process if we have hours in this day
                # Temperature: Create daily curve
                hourly_data.loc[mask, 'temperature'] = day['temperature'] - 2 * np.cos(
                    2 * np.pi * (day_hours - 14) / 24)

                # Rain: Distribute across random hours if there was rain
                if day['rain'] > 0:
                    n_rain_hours = min(8, len(day_hours))
                    rain_hours = np.random.choice(day_hours, size=n_rain_hours, replace=False)
                    rain_per_hour = day['rain'] / n_rain_hours

                    for hour in rain_hours:
                        hour_mask = mask & (hourly_data.index.hour == hour)
                        hourly_data.loc[hour_mask, 'rain'] = rain_per_hour

                hourly_data.loc[mask, 'rain'] = hourly_data.loc[mask, 'rain'].fillna(0)

                # Wind speed: Add slight variations
                hourly_data.loc[mask, 'wind_speed'] = day['wind_speed'] + np.random.normal(0, 0.5, size=len(day_hours))

                # Cloud cover: Keep constant
                hourly_data.loc[mask, 'cloud_cover'] = day['cloud_cover']

        # Reset index to make timestamp a column
        hourly_data = hourly_data.reset_index()
        hourly_data.columns = ['timestamp', 'temperature', 'rain', 'wind_speed', 'cloud_cover']

        # Fill any remaining NaN values with interpolation
        for col in ['temperature', 'wind_speed', 'cloud_cover']:
            hourly_data[col] = hourly_data[col].interpolate(method='linear')
        hourly_data['rain'] = hourly_data['rain'].fillna(0)

        print(f"Created {len(hourly_data)} hourly records")
        return hourly_data