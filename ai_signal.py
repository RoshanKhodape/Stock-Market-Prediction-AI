def get_ai_signal(current_price, predicted_price):

    current_price = float(current_price)
    predicted_price = float(predicted_price)

    if predicted_price > current_price:
        return "BUY"
    elif predicted_price < current_price:
        return "SELL"
    else:
        return "HOLD"