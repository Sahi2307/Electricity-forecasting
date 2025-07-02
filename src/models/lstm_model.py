import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

def train_lstm(config):
    # Load processed data
    processed_path = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    df = pd.read_csv(processed_path)

    # Features and target
    features = [
        'RRP', 'demand_pos_RRP', 'RRP_positive', 'demand_neg_RRP',
        'RRP_negative', 'frac_at_neg_RRP', 'min_temperature',
        'max_temperature', 'solar_exposure', 'rainfall',
        'school_day', 'holiday', 'day', 'month', 'year'
    ]
    target = 'demand'

    X = df[features].values
    y = df[target].values

    # Train/test split
    split_idx = int(len(df) * (1 - config['preprocessing']['test_size']))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for LSTM [samples, time_steps, features]
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build model
    model = Sequential()
    model.add(Input(shape=(1, X_train.shape[1])))
    model.add(LSTM(config['models']['lstm']['lstm_units'], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    model.fit(X_train_reshaped, y_train, epochs=config['models']['lstm']['epochs'], batch_size=config['models']['lstm']['batch_size'], verbose=1)

    # Save model
    model_path = os.path.join(config['paths']['models'], 'lstm_model.h5')
    os.makedirs(config['paths']['models'], exist_ok=True)
    model.save(model_path)

    # Evaluate
    y_pred = model.predict(X_test_reshaped).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"LSTM R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return model, (r2, mae, rmse) 