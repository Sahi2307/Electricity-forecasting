import pandas as pd
import os

def load_raw_data(config):
    """Load the raw dataset from the configured path."""
    data_path = config['data']['raw_data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    return df 