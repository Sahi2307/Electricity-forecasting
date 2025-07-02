import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

def preprocess_data(df, config):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Basic info (optional: print or log)
    # print(df.info())
    # print(df.describe())

    # Handle missing values (numeric: mean)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Scale all numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical variables
    le = LabelEncoder()
    df['school_day'] = le.fit_transform(df['school_day'])
    df['holiday'] = le.fit_transform(df['holiday'])

    # Extract additional date features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.day_name()

    # Save processed data
    processed_path = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    return df, scaler, le 