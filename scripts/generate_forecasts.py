import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta

def forecast_lstm_only(config, forecast_days=30):
    processed_path = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    df = pd.read_csv(processed_path)
    features = [
        'RRP', 'demand_pos_RRP', 'RRP_positive', 'demand_neg_RRP',
        'RRP_negative', 'frac_at_neg_RRP', 'min_temperature',
        'max_temperature', 'solar_exposure', 'rainfall',
        'school_day', 'holiday', 'day', 'month', 'year'
    ]
    last_row = df.iloc[-1].copy()
    forecasts_lstm = []
    dates = []
    lstm = load_model(os.path.join(config['paths']['models'], 'lstm_model.h5'))
    X_last = last_row[features].astype(float).values.reshape(1, -1)
    X_last_lstm = X_last.reshape((1, 1, len(features)))
    last_date = pd.to_datetime(last_row['date'])
    for i in range(forecast_days):
        lstm_pred = lstm.predict(X_last_lstm).flatten()[0]
        forecasts_lstm.append(lstm_pred)
        next_date = last_date + timedelta(days=1)
        dates.append(next_date)
        last_row['day'] = next_date.day
        last_row['month'] = next_date.month
        last_row['year'] = next_date.year
        last_row['date'] = next_date
        X_last = last_row[features].astype(float).values.reshape(1, -1)
        X_last_lstm = X_last.reshape((1, 1, len(features)))
        last_date = next_date
    return dates, forecasts_lstm

def plot_lstm_forecast(dates, lstm_forecast, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, lstm_forecast, marker='o', color='blue', label='Predicted Demand')
    plt.title('One-Month Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Demand')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    forecast_dir = os.path.join(config['paths']['figures'], 'forecasts')
    os.makedirs(forecast_dir, exist_ok=True)
    dates, lstm_forecast = forecast_lstm_only(config, forecast_days=30)
    plot_lstm_forecast(dates, lstm_forecast, save_path=os.path.join(forecast_dir, 'one_month_forecast.png'))
    print(f"Forecast plot saved to {os.path.join(forecast_dir, 'one_month_forecast.png')}")

if __name__ == '__main__':
    main() 