def generate_final_signal(tech_signal, ai_signal, sentiment_signal="Neutral", fundamental_data=None):
    """
    Institutional Scoring Matrix for High Probability Signals.
    Max Possible Score: +6 (Bullish) | Min Possible Score: -6 (Bearish)
    """
    score = 0

    # 1. Technical Analysis Alignment (Weight: Max 2)
    if tech_signal == "STRONG BUY":
        score += 2
    elif tech_signal == "BUY":
        score += 1
    elif tech_signal == "SELL":
        score -= 1
    elif tech_signal == "STRONG SELL":
        score -= 2

    # 2. AI Prediction Alignment (Weight: Max 2)
    if ai_signal == "BUY":
        score += 2  
    elif ai_signal == "SELL":
        score -= 2

    # 3. Sentiment Data Filter (Weight: Max 1)
    if sentiment_signal == "Positive":
        score += 1
    elif sentiment_signal == "Negative":
        score -= 1

    # 4. Fundamental Health Filter (Weight: Max 1)
    # Checks if the company is profitable (EPS > 0) and reasonably valued
    if fundamental_data and isinstance(fundamental_data, dict):
        eps = fundamental_data.get("eps")
        pe = fundamental_data.get("pe")
        
        if eps and float(eps) > 0:
            # Healthy earnings gets a boost if buying
            if tech_signal in ["BUY", "STRONG BUY"] and pe and float(pe) < 40:
                score += 1
            # Unhealthy or highly overvalued structures degrade short sell blocks less
            elif tech_signal in ["SELL", "STRONG SELL"] and pe and float(pe) > 80:
                score -= 1

    # --- FINAL HYBRID EXECUTION MATRIX ---
    # High-probability signals require a minimum score of 4 out of 6
    if score >= 4:
        return "HIGH PROBABILITY BUY"
    elif score == 3:
        return "BUY"
    elif score <= -4:
        return "HIGH PROBABILITY SELL"
    elif score == -3:
        return "SELL"
    else:
        return "WAIT / HOLD"