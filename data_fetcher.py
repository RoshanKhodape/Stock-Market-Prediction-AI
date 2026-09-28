import yfinance as yf

def get_stock_data(symbol):
    df = yf.download(symbol, start="2018-01-01")
    return df