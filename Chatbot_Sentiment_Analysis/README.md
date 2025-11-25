# Chatbot with Sentiments analysis Project

**What this contains**
- A CLI chatbot that stores conversation history and performs sentiment analysis.
- Hybrid sentiment engine (VADER + TextBlob) with comparative-aware weighting.
- Improved conversation-level sentiment logic that biases toward negative signals
  and discounts comparative false positives.
- Tests and instructions.

**How to run**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the chatbot:
   ```bash
   python app.py
   ```
   Type messages. Use `exit` or `quit` to finish and see the sentiment report.

## Sentiment logic
- **Statement-level :** Each user message is scored with VADER producing a `compound` value in [-1,1].
  - `compound >= 0.05` => Positive
  - `compound <= -0.05` => Negative
  - otherwise => Neutral
- **Conversation-level :** Average the compound scores of all user messages to determine overall sentiment using the same thresholds. Also report counts and provide an overall direction and magnitude.

## Status 
- Statement-level analysis (per user message) implemented and printed at the end of the session.
- Optional trend summary (shift detection) included: shows whether conversation sentiment moved upward or downward overall.

## Files of interest
- `src/chatbot/sentiment.py` : sentiment utilities
- `src/chatbot/chatbot.py` : Chatbot implementation (keeps conversation history)
- `app.py` : CLI runner that collects messages and prints results