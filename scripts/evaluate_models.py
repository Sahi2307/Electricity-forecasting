import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.evaluation.visualization import plot_actual_vs_pred, plot_year_vs_demand

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    processed_path = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    df = pd.read_csv(processed_path)
    features = [
        'RRP', 'demand_pos_RRP', 'RRP_positive', 'demand_neg_RRP',
        'RRP_negative', 'frac_at_neg_RRP', 'min_temperature',
        'max_temperature', 'solar_exposure', 'rainfall',
        'school_day', 'holiday', 'day', 'month', 'year'
    ]
    target = 'demand'
    split_idx = int(len(df) * (1 - config['preprocessing']['test_size']))
    X_test = df[features].iloc[split_idx:].values
    y_test = df[target].iloc[split_idx:].values
    df_test = df.iloc[split_idx:].copy()

    # Random Forest
    rf = joblib.load(os.path.join(config['paths']['models'], 'random_forest_model.pkl'))
    rf_pred = rf.predict(X_test)
    perf_dir = os.path.join(config['paths']['figures'], 'model_performance')
    os.makedirs(perf_dir, exist_ok=True)
    plot_actual_vs_pred(y_test, rf_pred, 'Random Forest', save_path=os.path.join(perf_dir, 'rf_actual_vs_pred.png'))
    plot_year_vs_demand(df_test, rf_pred, 'Random Forest', save_path=os.path.join(perf_dir, 'rf_year_vs_demand.png'))

    # LSTM
    lstm = load_model(os.path.join(config['paths']['models'], 'lstm_model.h5'))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_pred = lstm.predict(X_test_lstm).flatten()
    plot_actual_vs_pred(y_test, lstm_pred, 'LSTM', save_path=os.path.join(perf_dir, 'lstm_actual_vs_pred.png'))
    plot_year_vs_demand(df_test, lstm_pred, 'LSTM', save_path=os.path.join(perf_dir, 'lstm_year_vs_demand.png'))

if __name__ == '__main__':
    main() 