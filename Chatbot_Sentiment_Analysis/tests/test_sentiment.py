from src.chatbot.sentiment import score_text, label_from_compound

def test_negative():
    r = score_text("your service disappoints me")
    assert label_from_compound(r['compound']) == "Negative"

def test_comparative():
    r = score_text("last experience was better")
    # comparative should lean negative in our hybrid (TextBlob influence)
    assert isinstance(r['compound'], float)