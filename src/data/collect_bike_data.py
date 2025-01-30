import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class BikeDataCollector:
    def __init__(self):
        # Define Amsterdam locations
        self.locations = [
            {'id': 1, 'name': 'Central Station', 'latitude': 52.3791, 'longitude': 4.9003},
            {'id': 2, 'name': 'Vondelpark', 'latitude': 52.3579, 'longitude': 4.8686},
            {'id': 3, 'name': 'Dam Square', 'latitude': 52.3731, 'longitude': 4.8933}
        ]

    def get_bike_counts(self, start_date, end_date):
        """Generate synthetic bike count data"""
        # Calculate number of days
        days = (end_date - start_date).days + 1

        # Create date range
        dates = pd.date_range(start=start_date, periods=days * 24, freq='H')

        # Base data
        df = pd.DataFrame({'timestamp': dates})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # Define typical patterns
        base_counts = {
            'weekday_rush': 500,
            'weekday_normal': 200,
            'weekend_peak': 300,
            'weekend_normal': 150
        }

        # Generate counts
        counts = []
        for _, row in df.iterrows():
            hour = row['hour']
            is_weekend = row['day_of_week'] >= 5

            # Determine base count
            if is_weekend:
                base = base_counts['weekend_peak'] if 11 <= hour <= 16 else base_counts['weekend_normal']
            else:
                if hour in [8, 9, 17, 18]:  # Rush hours
                    base = base_counts['weekday_rush']
                else:
                    base = base_counts['weekday_normal']

            # Add variation
            variation = np.random.uniform(0.8, 1.2)
            count = int(base * variation)

            # Reduce for early morning
            if 0 <= hour <= 5:
                count = int(count * 0.2)

            counts.append(count)

        df['count'] = counts

        # Generate for multiple locations
        all_data = []
        for loc in self.locations:
            df_loc = df.copy()
            df_loc['location_id'] = loc['id']
            # Adjust counts based on location
            if loc['id'] == 1:  # Central Station
                df_loc['count'] = (df_loc['count'] * 1.2).astype(int)
            elif loc['id'] == 2:  # Vondelpark
                df_loc['count'] = (df_loc['count'] * 0.9).astype(int)
            all_data.append(df_loc)

        return pd.concat(all_data, ignore_index=True)

    def get_locations(self):
        """Get location data"""
        return pd.DataFrame(self.locations)

    def save_data(self, data, filename):
        """Save data to CSV file"""
        if data is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            data.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")


if __name__ == "__main__":
    # Example usage
    collector = BikeDataCollector()

    # Get data for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Generate and save bike count data
    bike_data = collector.get_bike_counts(start_date, end_date)
    collector.save_data(bike_data, 'data/raw/bike_counts.csv')

    # Save location data
    locations = collector.get_locations()
    collector.save_data(locations, 'data/raw/locations.csv')