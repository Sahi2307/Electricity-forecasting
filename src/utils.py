import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    features = ['demand', 'price']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler
