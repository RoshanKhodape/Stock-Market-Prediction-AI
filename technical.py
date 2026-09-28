import numpy as np
import pandas as pd

def calculate_adx(df, period=14):
    """Calculates the Average Directional Index (ADX) to determine market regime strength."""
    df = df.copy()
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    
    # Smooth wild daily ranges using Wilders smoothing approach
    tr_smooth = df['TR'].rolling(period).sum()
    plus_dm_smooth = df['+DM'].rolling(period).sum()
    minus_dm_smooth = df['-DM'].rolling(period).sum()
    
    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)
    
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    adx = df['DX'].rolling(period).mean()
    
    return float(adx.iloc[-1])

def get_technical_signal(df):
    df = df.copy()

    # 1. Base Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    df = df.dropna()
    if df.empty:
        return "HOLD"

    latest = df.iloc[-1]
    
    price = float(latest['Close'])
    sma20 = float(latest['SMA_20'])
    sma50 = float(latest['SMA_50'])
    sma200 = float(latest['SMA_200'])
    rsi = float(latest['RSI'])
    macd = float(latest['MACD'])
    macd_sig = float(latest['MACD_Signal'])
    bb_lower = float(latest['BB_Lower'])
    bb_upper = float(latest['BB_Upper'])

    # Determine trend strength using our helper function
    adx_value = calculate_adx(df)
    is_trending = adx_value > 23

    # --- ADVANCED REGIME SELECTOR LOGIC ---

    if is_trending:
        # REGIME A: TRENDING MARKET (Trust Crossovers and Momentum)
        is_bullish_trend = price > sma200 and sma20 > sma50
        is_bearish_trend = price < sma200 and sma20 < sma50
        
        buy_momentum = rsi > 55 and macd > macd_sig
        sell_momentum = rsi < 45 and macd < macd_sig

        if is_bullish_trend and buy_momentum:
            if price < bb_upper: # Don't chase at the absolute ceiling band
                return "STRONG BUY"
            return "BUY"
            
        elif is_bearish_trend and sell_momentum:
            if price > bb_lower: # Don't short into the absolute floor band
                return "STRONG SELL"
            return "SELL"

    else:
        # REGIME B: RANGING/SIDEWAYS MARKET (Trade Mean Reversion)
        # In a sideways market, high RSI means exhausted buyers, low RSI means exhausted sellers.
        if price <= bb_lower or rsi < 30:
            return "BUY"  # Price hit structural channel floor
        elif price >= bb_upper or rsi > 70:
            return "SELL" # Price hit structural channel ceiling

    return "HOLD"