import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR) using Wilders True Range formula.
    Ensures stop loss settings adjust perfectly to current market volatility.
    """
    df = df.copy()
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    
    # True Range is the maximum value among these three metrics
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Use exponential rolling structure for smoother volatility adjustment
    atr_series = df['TR'].ewm(span=period, adjust=False).mean()
    
    return float(atr_series.iloc[-1])

def calculate_risk(df, entry_price, signal):
    """
    Calculates statistical Stop Loss and Target levels based on asset volatility.
    Targets a 1:2 Risk-to-Reward Ratio (R:R) for strict account growth.
    """
    entry_price = float(entry_price)
    atr = calculate_atr(df)
    
    # Volatility Multiplier (Positions risk outside daily market noise)
    multiplier = 2.0
    
    if signal in ["BUY", "STRONG BUY", "HIGH PROBABILITY BUY"]:
        stop_loss = round(entry_price - (atr * multiplier), 2)
        target = round(entry_price + (atr * multiplier * 2.0), 2)
        
    elif signal in ["SELL", "STRONG SELL", "HIGH PROBABILITY SELL"]:
        stop_loss = round(entry_price + (atr * multiplier), 2)
        target = round(entry_price - (atr * multiplier * 2.0), 2)
        
    else:
        return None, None, None

    risk = abs(entry_price - stop_loss)
    reward = abs(target - entry_price)
    
    # Risk-to-reward ratio metric calculation
    rr_ratio = round(reward / risk, 2) if risk != 0 else 0
    
    return stop_loss, target, rr_ratio