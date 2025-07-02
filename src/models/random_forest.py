import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

def train_random_forest(config):
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

    X = df[features]
    y = df[target]

    # Train/test split
    split_idx = int(len(df) * (1 - config['preprocessing']['test_size']))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    rf = RandomForestRegressor(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth'],
        random_state=config['models']['random_forest']['random_state']
    )
    rf.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(config['paths']['models'], 'random_forest_model.pkl')
    os.makedirs(config['paths']['models'], exist_ok=True)
    joblib.dump(rf, model_path)

    # Evaluate
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return rf, (r2, mae, rmse) 