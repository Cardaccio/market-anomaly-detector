import pandas as pd
import numpy as np

def detect_rsi_anomalies(df, low=20, high=80):
    df['rsi_anomaly'] = (df['rsi'] > high) | (df['rsi'] < low)
    return df

def detect_bollinger_anomalies(df):
    df['bb_anomaly'] = (df['Close'] > df['bb_h']) | (df['Close'] < df['bb_l'])
    return df

def detect_return_anomalies(df, threshold=3):
    df['log_return_z'] = df.groupby('Ticker')['log_return'].transform(lambda x: (x - x.mean()) / x.std())
    df['return_anomaly'] = df['log_return_z'].abs() > threshold
    return df

def detect_all_anomalies(df):
    df = detect_rsi_anomalies(df)
    df = detect_bollinger_anomalies(df)
    df = detect_return_anomalies(df)
    return df
