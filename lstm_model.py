import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def compute_indicators(df):
    """Adds critical momentum and structural features to the dataframe."""
    df = df.copy()
    
    # 1. RSI Feature
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD Feature
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    df = df.dropna()
    return df

def train_and_predict(stock_symbol):
    model_path = os.path.join(MODEL_DIR, f"{stock_symbol}_multivariate.keras")
    
    # Fetch historical data
    df = yf.download(stock_symbol, period="5y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.capitalize() for col in df.columns]
    
    # Compute complex indicator data matrix
    df = compute_indicators(df)
    
    # Select our 4 key features for multivariate learning
    feature_cols = ['Close', 'Volume', 'RSI', 'MACD']
    data_matrix = df[feature_cols].values
    
    # Scale each feature independently between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_matrix)
    
    # Target scale setup specifically for decoding the Close price later
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[['Close']].values)

    lookback = 60

    if os.path.exists(model_path):
        print(f"--- Loading existing Multivariate model for {stock_symbol} ---")
        model = load_model(model_path)
    else:
        print(f"--- Training NEW Multivariate model for {stock_symbol} ---")
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])       # Appends shape (60, 4)
            y.append(scaled_data[i, 0])              # Predicts the next Close price (Index 0)
            
        X, y = np.array(X), np.array(y)

        # Build deeper sequence network
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(lookback, len(feature_cols))),
            Dropout(0.2),
            LSTM(units=100, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=12, batch_size=32, verbose=0)
        
        model.save(model_path)
        print(f"--- Model saved at {model_path} ---")

    # Extract the last 60 rows across all 4 columns for accurate prediction tracking
    last_60_days = scaled_data[-lookback:]
    last_60_days = np.reshape(last_60_days, (1, lookback, len(feature_cols)))
    
    predicted_scaled = model.predict(last_60_days, verbose=0)
    predicted_price = close_scaler.inverse_transform(predicted_scaled)

    return round(float(predicted_price[0][0]), 2)