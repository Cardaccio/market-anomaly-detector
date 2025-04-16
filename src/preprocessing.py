# src/preprocessing.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_rsi(df, window=14):
    df['rsi'] = df.groupby('Ticker')['Close'].transform(
        lambda x: RSIIndicator(x, window=window).rsi()
    )
    return df

def add_bollinger_bands(df, window=20):
    def calc_bb(x):
        bb = BollingerBands(close=x, window=window)
        return pd.DataFrame({
            'bb_mavg': bb.bollinger_mavg(),
            'bb_h': bb.bollinger_hband(),
            'bb_l': bb.bollinger_lband()
        }, index=x.index)

    bb_data = df.groupby('Ticker')['Close'].apply(calc_bb).reset_index(level=0, drop=True)
    df = df.join(bb_data)
    return df

def add_log_returns(df):
    df['log_return'] = df.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x) - np.log(x.shift(1))
    )
    return df

def add_technical_indicators(df):
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_log_returns(df)
    df.dropna(inplace=True)  # Remove first rows with NaNs
    return df
