import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import joblib
import yaml
import os


class BikeUsagePredictor:
    def __init__(self, config_path='../config/config.yaml'):
        """Initialize the predictor with configuration"""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using default settings.")
            self.config = {
                'model_params': {
                    'prophet': {
                        'yearly_seasonality': True,
                        'weekly_seasonality': True,
                        'daily_seasonality': True,
                        'seasonality_mode': 'multiplicative',
                        'uncertainty_samples': 1000
                    }
                }
            }

        self.model = Prophet(**self.config['model_params']['prophet'])
        self.regressor_defaults = None

    def prepare_data(self, df):
        """Prepare data for Prophet model"""
        prophet_df = df.copy()

        # Store mean values of regressors for future predictions
        regressor_columns = ['temperature', 'rain', 'wind_speed', 'is_weekend', 'is_rush_hour']
        self.regressor_defaults = {col: df[col].mean() for col in regressor_columns if col in df.columns}

        # Rename columns to Prophet requirements
        prophet_df['ds'] = pd.to_datetime(prophet_df['timestamp'])
        prophet_df['y'] = prophet_df['count']

        return prophet_df

    def add_regressors(self):
        """Add additional regressors to the Prophet model"""
        if self.regressor_defaults:
            for regressor in self.regressor_defaults.keys():
                self.model.add_regressor(regressor)

    def train(self, df, add_regressors=True):
        """Train the Prophet model"""
        print("Preparing data for training...")
        prophet_df = self.prepare_data(df)

        if add_regressors:
            print("Adding regressors...")
            self.add_regressors()
            # Add regressor values to prophet_df
            for column in self.regressor_defaults.keys():
                prophet_df[column] = df[column]

        print("Training model...")
        self.model.fit(prophet_df)
        print("Model training completed!")

    def predict(self, periods=24, freq='H', future_regressors=None):
        """Make predictions with the trained model"""
        print(f"Generating predictions for next {periods} {freq}...")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Add default regressor values if not provided
        if not future_regressors and self.regressor_defaults:
            future_regressors = {
                regressor: [value] * len(future)
                for regressor, value in self.regressor_defaults.items()
            }

        if future_regressors:
            for regressor, values in future_regressors.items():
                if isinstance(values, (int, float)):
                    future[regressor] = values
                else:
                    future[regressor] = values[:len(future)]

        forecast = self.model.predict(future)

        # Add prediction intervals
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

        return forecast

    def evaluate(self, df):
        """
        Evaluate model performance
        """
        # Prepare data
        prophet_df = self.prepare_data(df)
        
        # Make predictions for the same timestamps as in the training data
        future = prophet_df[['ds']].copy()
        
        # Add regressor values from training data
        if self.regressor_defaults:
            for regressor, _ in self.regressor_defaults.items():
                future[regressor] = prophet_df[regressor]
        
        # Get predictions
        forecast = self.model.predict(future)
        
        # Get actual and predicted values
        y_true = prophet_df['y'].values
        y_pred = forecast['yhat'].values
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def save_model(self, path='models/saved_models/prophet_model.joblib'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path='models/saved_models/prophet_model.joblib'):
        """Load a trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")