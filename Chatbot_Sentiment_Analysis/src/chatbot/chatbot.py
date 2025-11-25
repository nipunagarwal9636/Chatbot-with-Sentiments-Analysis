from typing import List, Dict, Optional, Callable
from .sentiment import score_text, label_from_compound

class Chatbot:
    def __init__(self, llm: Optional[Callable[[str], str]] = None):
        self.history: List[Dict] = []
        self.llm = llm

    def add_user_message(self, text: str):
        self.history.append({'speaker': 'user', 'text': text})

    def add_bot_message(self, text: str):
        self.history.append({'speaker': 'bot', 'text': text})

    def get_response(self, user_text: str) -> str:
        # LLM optional; fallback canned responses
        if self.llm:
            try:
                if callable(self.llm):
                    response = self.llm(user_text)
                else:
                    response = str(self.llm)
            except Exception:
                response = self._generate_canned_response(user_text)
        else:
            response = self._generate_canned_response(user_text)

        self.add_bot_message(response)
        return response

    def _generate_canned_response(self, user_text: str) -> str:
        text = user_text.lower()
        if any(x in text for x in ["help", "issue", "problem", "error", "fail"]):
            return "I'm sorry you're having trouble. Could you tell me more about the issue?"
        if "price" in text or "cost" in text:
            return "I don't have pricing info here, but I can help you find it or escalate the question."
        if any(x in text for x in ["thanks", "thank you", "thx"]):
            return "You're welcome! Glad I could help."
        return "Thanks for telling me. I hear you."

    def user_messages(self) -> List[str]:
        return [m['text'] for m in self.history if m['speaker']=='user']

    def conversation_sentiment_report(self) -> str:
        user_msgs = self.user_messages()
        if not user_msgs:
            return "No user messages to analyze."

        per_message = []
        for m in user_msgs:
            s = score_text(m)
            label = label_from_compound(s['compound'])
            per_message.append({'text': m, 'scores': s, 'label': label})

        lines = []
        lines.append("=== Statement-level Sentiment ===")
        pos = neg = neu = 0
        for i, pm in enumerate(per_message, 1):
            if pm['label'] == 'Positive':
                pos += 1
            elif pm['label'] == 'Negative':
                neg += 1
            else:
                neu += 1
            lines.append(f"{i}. \"{pm['text']}\" -> {pm['label']} (compound={pm['scores']['compound']:.3f})")

        
        overall = self._overall_conversation_sentiment(per_message)

        lines.append("\n=== Conversation-level Sentiment ===")
        lines.append(f"Messages analyzed: {len(per_message)}")
        lines.append(f"Positive: {pos}, Neutral: {neu}, Negative: {neg}")
        avg_compound = sum(p['scores']['compound'] for p in per_message)/len(per_message)
        lines.append(f"Average compound score: {avg_compound:.3f}")
        lines.append(f"Overall conversation sentiment: {overall}")

        compounds = [p['scores']['compound'] for p in per_message]
        half = len(compounds)//2
        first_avg = sum(compounds[:half])/max(1, len(compounds[:half]))
        second_avg = sum(compounds[half:])/max(1, len(compounds[half:]))
        trend = "no significant change"
        if second_avg - first_avg >= 0.05:
            trend = "improving (more positive over time)"
        elif first_avg - second_avg >= 0.05:
            trend = "worsening (more negative over time)"
        lines.append(f"Trend across conversation: {trend}")

        return "\n".join(lines)

    def _overall_conversation_sentiment(self, per_message: List[Dict]) -> str:
        labels = [p['label'] for p in per_message]
        texts = [p['text'] for p in per_message]
        pos_count = labels.count('Positive')
        neg_count = labels.count('Negative')

        weighted_score = (pos_count * 1.0) - (neg_count * 1.5)

        if labels and labels[0] == 'Negative':
            weighted_score -= 0.5

        comparative_keywords = ["better", "worse", "less", "more", "than"] 
        for i, txt in enumerate(texts):
            if any(k in txt.lower() for k in comparative_keywords):
                if labels[i] == 'Positive':
                    weighted_score -= 0.4

        if weighted_score < 0:
            return 'Negative - general dissatisfaction'
        elif weighted_score > 0:
            return 'Positive'
        else:
            return 'Neutral'
