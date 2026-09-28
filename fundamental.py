import yfinance as yf

def get_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info:
            return {"pe": None, "eps": None, "market_cap": None}
            
        return {
            "pe": info.get("trailingPE") or info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap")
        }
    except Exception:
        return {"pe": None, "eps": None, "market_cap": None}