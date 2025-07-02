import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.models.random_forest import train_random_forest
from src.models.lstm_model import train_lstm

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('Training Random Forest...')
    train_random_forest(config)
    print('Training LSTM...')
    train_lstm(config)

if __name__ == '__main__':
    main() 