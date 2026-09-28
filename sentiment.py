from textblob import TextBlob

def get_sentiment(headlines):
    score = 0
    for news in headlines:
        score += TextBlob(news).sentiment.polarity

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"