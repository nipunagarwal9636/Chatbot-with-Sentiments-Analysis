from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

vader = SentimentIntensityAnalyzer()


def is_comparative(text: str) -> bool:
    keywords = [
        "better", "worse", "less", "more", "improved", "declined",
        "than before", "than earlier", "than last time", "better than", "worse than"
    ]
    t = text.lower()
    return any(k in t for k in keywords)


def score_text(text: str) -> dict:
    # VADER scores
    v_scores = vader.polarity_scores(text)
    v_compound = v_scores.get("compound", 0.0)

    tb_polarity = TextBlob(text).sentiment.polarity  # -1 -> 1

    vader_w = 0.4
    textblob_w = 0.6
    if is_comparative(text):
        vader_w = 0.2
        textblob_w = 0.8

    compound = (vader_w * v_compound) + (textblob_w * tb_polarity)

    pos = max(0.0, compound)
    neg = max(0.0, -compound)
    neu = 1.0 - (pos + neg)

    return {
        "text": text,
        "neg": neg,
        "neu": neu,
        "pos": pos,
        "compound": compound
    }


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"
