import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


class BikeUsageVisualizer:
    def __init__(self, config_path='config/config.yaml'):
        self.amsterdam_coords = [52.3676, 4.9041]
        plt.style.use('seaborn')

    def plot_daily_usage(self, df, save_path=None):
        """Plot daily bike usage patterns"""
        plt.figure(figsize=(12, 6))
        daily_usage = df.groupby(pd.to_datetime(df['timestamp']).dt.date)['count'].mean()

        plt.plot(daily_usage.index, daily_usage.values)
        plt.title('Daily Bike Usage Pattern')
        plt.xlabel('Date')
        plt.ylabel('Average Bike Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_hourly_heatmap(self, df, save_path=None):
        """Create heatmap of hourly usage patterns by day of week"""
        plt.figure(figsize=(12, 8))

        # Prepare data for heatmap
        hourly_usage = df.pivot_table(
            values='count',
            index=pd.to_datetime(df['timestamp']).dt.hour,
            columns=pd.to_datetime(df['timestamp']).dt.day_name(),
            aggfunc='mean'
        )

        # Create heatmap
        sns.heatmap(
            hourly_usage,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Average Bike Count'}
        )

        plt.title('Hourly Bike Usage by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def create_location_map(self, df, locations_df, save_path=None):
        """Create an interactive map with bike usage by location"""
        # Create base map
        m = folium.Map(
            location=self.amsterdam_coords,
            zoom_start=13,
            tiles='CartoDB positron'
        )

        # Add location markers with bike usage info
        for _, location in locations_df.iterrows():
            location_data = df[df['location_id'] == location['id']]
            avg_usage = location_data['count'].mean()

            folium.CircleMarker(
                location=[location['latitude'], location['longitude']],
                radius=np.sqrt(avg_usage) / 2,  # Scale radius by square root of usage
                popup=f"Location ID: {location['id']}<br>Average Usage: {avg_usage:.0f}",
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)

        if save_path:
            m.save(save_path)

        return m

    def plot_weather_correlation(self, df, save_path=None):
        """Plot correlation between weather factors and bike usage"""
        plt.figure(figsize=(10, 8))

        weather_cols = ['temperature', 'rain', 'wind_speed', 'count']
        correlation_matrix = df[weather_cols].corr()

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0
        )

        plt.title('Correlation between Weather Factors and Bike Usage')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_predictions_vs_actual(self, forecast_df, actual_df, save_path=None):
        """Plot model predictions against actual values"""
        plt.figure(figsize=(15, 7))

        plt.plot(actual_df['timestamp'], actual_df['count'], label='Actual', alpha=0.7)
        plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Predicted', alpha=0.7)
        plt.fill_between(
            forecast_df['ds'],
            forecast_df['yhat_lower'],
            forecast_df['yhat_upper'],
            alpha=0.3,
            label='95% Confidence Interval'
        )

        plt.title('Predicted vs Actual Bike Usage')
        plt.xlabel('Date')
        plt.ylabel('Bike Count')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def create_interactive_dashboard(self, df, forecast_df=None):
        """Create an interactive Plotly dashboard"""
        # Daily usage trend
        daily_fig = go.Figure()
        daily_fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['count'],
            name='Actual Usage',
            mode='lines'
        ))

        if forecast_df is not None:
            daily_fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                name='Predicted Usage',
                mode='lines'
            ))

        daily_fig.update_layout(
            title='Bike Usage Over Time',
            xaxis_title='Date',
            yaxis_title='Bike Count'
        )

        # Weather impact
        weather_fig = px.scatter(
            df,
            x='temperature',
            y='count',
            color='rain',
            size='wind_speed',
            title='Impact of Weather on Bike Usage'
        )

        return daily_fig, weather_fig


if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('data/processed/featured_data.csv')
        locations_df = pd.read_csv('data/raw/locations.csv')

        # Initialize visualizer
        visualizer = BikeUsageVisualizer()

        # Create visualizations
        visualizer.plot_daily_usage(df, 'reports/figures/daily_usage.png')
        visualizer.plot_hourly_heatmap(df, 'reports/figures/hourly_heatmap.png')
        visualizer.plot_weather_correlation(df, 'reports/figures/weather_correlation.png')
        visualizer.create_location_map(df, locations_df, 'reports/figures/location_map.html')

        print("Visualizations created successfully!")

    except Exception as e:
        print(f"Error creating visualizations: {e}")