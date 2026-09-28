from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from lstm_model import train_and_predict
from technical import get_technical_signal
from ai_signal import get_ai_signal
from final_signal import generate_final_signal
from risk_management import calculate_risk
from backtest import backtest_strategy
from fundamental import get_fundamentals
from sentiment import get_sentiment

app = Flask(__name__)

def fetch_stock_data(stock):
    """Stock data download ani clean karnyasaathi robust function."""
    try:
        df = yf.download(stock, period="5y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.capitalize() for col in df.columns]
        if len(df) < 100:
            return None
        return df.dropna()
    except Exception as e:
        print(f"DEBUG Error: {str(e)}")
        return None

def fetch_news_headlines(stock):
    """Yahoo Finance varun news headlines kaadhnya sathi function."""
    try:
        ticker = yf.Ticker(stock)
        news = ticker.news
        headlines = [item['title'] for item in news if 'title' in item]
        return headlines[:5]  # Fakt top 5 headlines gheu
    except Exception as e:
        print(f"DEBUG News Error: {str(e)}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    # 🔹 1. Default context setup
    context = {
        "stock": "^NSEI",
        "predicted": None,
        "error": None,
        "ohlc_data": [],
        "volume_data": [],
        "rsi_data": [],
        "final_signal": None,
        "current_price": None,
        "ai_signal": "HOLD",
        "tech_signal": "HOLD",
        "sentiment_signal": "Neutral",
        "stop_loss": None,
        "target": None,
        "rr_ratio": None,
        "backtest_result": None  
    }

    if request.method == 'POST':
        context["stock"] = request.form.get('stock', '^NSEI').upper().strip()
        stock = context["stock"]

        try:
            if stock == "^NSEFIN":
                raise ValueError("Selected index not supported by Yahoo Finance.")

            df = fetch_stock_data(stock)
            if df is None:
                raise ValueError(f"No data found or insufficient history for {stock}.")

            # 🔹 RSI Calculation safely
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df = df.dropna()

            # 🔹 Chart Data Preparation
            for i in range(len(df)):
                context["ohlc_data"].append({
                    "x": i, "o": round(float(df['Open'].iloc[i]), 2),
                    "h": round(float(df['High'].iloc[i]), 2),
                    "l": round(float(df['Low'].iloc[i]), 2),
                    "c": round(float(df['Close'].iloc[i]), 2)
                })
                context["volume_data"].append({"x": i, "y": float(df['Volume'].iloc[i])})
                context["rsi_data"].append({"x": i, "y": round(float(df['RSI'].iloc[i]), 2)})

            # 🔹 AI Prediction & Current Price
            context["predicted"] = round(train_and_predict(stock), 2)
            context["current_price"] = round(float(df['Close'].iloc[-1]), 2)

            # AI Signal Matrix Logic
            curr, pred = context["current_price"], context["predicted"]
            threshold = 0.005  
            if pred > curr * (1 + threshold): context["ai_signal"] = "BUY"
            elif pred < curr * (1 - threshold): context["ai_signal"] = "SELL"
            else: context["ai_signal"] = "HOLD"

            # 🔹 New Filters: Fundamentals ani Sentiment Fetch kara
            fund_data = get_fundamentals(stock)
            headlines = fetch_news_headlines(stock)
            context["sentiment_signal"] = get_sentiment(headlines)

            # 🔹 Technical Signal Engine
            context["tech_signal"] = get_technical_signal(df)
            
            # 🔹 Combine all 4 layers for Final Master Signal
            context["final_signal"] = generate_final_signal(
                tech_signal=context["tech_signal"], 
                ai_signal=context["ai_signal"], 
                sentiment_signal=context["sentiment_signal"], 
                fundamental_data=fund_data
            )

            # 🔹 Risk Mitigation (Stop Loss & Target)
            if context["final_signal"] not in ["HOLD", "WAIT / HOLD"]:
                sl, tgt, rr = calculate_risk(df, context["current_price"], context["final_signal"])
                context.update({"stop_loss": sl, "target": tgt, "rr_ratio": rr})

            # 🔹 Accuracy/Backtest Evaluation Data
            context["backtest_result"] = backtest_strategy(stock)

        except Exception as e:
            context["error"] = str(e)
            print(f"Execution Error: {e}") 

    return render_template("index.html", **context)

@app.route('/live_price/<symbol>')
def live_price(symbol):
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data.empty: return jsonify({"error": "No data"})
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.capitalize() for col in data.columns]
        return jsonify({"price": round(float(data['Close'].iloc[-1]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)