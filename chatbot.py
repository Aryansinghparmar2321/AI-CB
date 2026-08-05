import json
import re
import string
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CustomerServiceChatbot:
    def __init__(self, intents_path="intents.json", similarity_threshold=0.35):
        self.intents_path = Path(intents_path)
        self.similarity_threshold = similarity_threshold
        self.intents = self._load_intents()

        # Flatten (pattern -> tag) for both the keyword layer and the NLP layer
        self.pattern_texts = []
        self.pattern_tags = []
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                self.pattern_texts.append(pattern)
                self.pattern_tags.append(intent["tag"])

        # Fit TF-IDF over all known patterns once, at startup
        self.vectorizer = TfidfVectorizer(preprocessing=None) if False else TfidfVectorizer()
        self.pattern_vectors = self.vectorizer.fit_transform(
            [self._clean(t) for t in self.pattern_texts]
        )

        self.tag_to_responses = {
            intent["tag"]: intent["responses"] for intent in self.intents["intents"]
        }
        self.fallback_responses = self.intents.get(
            "fallback_responses", ["I'm not sure I understand — could you rephrase that?"]
        )
        self._fallback_i = 0

    # ------------------------------------------------------------------
    def _load_intents(self):
        with open(self.intents_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _clean(text):
        """Lowercase + strip punctuation for more robust matching."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    # ------------------------------------------------------------------
    def _rule_based_match(self, message):
        """
        Fast path: does the cleaned message contain a known pattern as
        a whole phrase (matched on word boundaries, not raw substring —
        otherwise short patterns like "hi" would wrongly match inside
        words like "this" or "shipping")?
        """
        cleaned = self._clean(message)
        message_words = set(cleaned.split())

        best_tag = None
        best_len = 0
        for pattern, tag in zip(self.pattern_texts, self.pattern_tags):
            cleaned_pattern = self._clean(pattern)
            # word-boundary substring match (handles multi-word patterns)
            if re.search(rf"\b{re.escape(cleaned_pattern)}\b", cleaned):
                if len(cleaned_pattern) > best_len:
                    best_tag, best_len = tag, len(cleaned_pattern)
                continue
            # whole-message == whole-pattern as a set of words (short greetings etc.)
            pattern_words = set(cleaned_pattern.split())
            if pattern_words and pattern_words == message_words:
                if len(cleaned_pattern) > best_len:
                    best_tag, best_len = tag, len(cleaned_pattern)

        return best_tag

    def _nlp_match(self, message):
        """
        Fuzzy path: TF-IDF + cosine similarity against every known
        pattern. Returns (tag, score) for the best match.
        """
        cleaned = self._clean(message)
        vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(vec, self.pattern_vectors)[0]
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        best_tag = self.pattern_tags[best_idx]
        return best_tag, best_score

    # ------------------------------------------------------------------
    def get_response(self, message):
        if not message or not message.strip():
            return "I didn't catch that — could you type your question?"

        # 1. Try exact rule-based match first
        tag = self._rule_based_match(message)
        matched_via = "rule"

        # 2. Fall back to NLP similarity match
        if tag is None:
            tag, score = self._nlp_match(message)
            matched_via = "nlp"
            if score < self.similarity_threshold:
                return self._fallback()

        import random
        response = random.choice(self.tag_to_responses[tag])
        return response

    def _fallback(self):
        response = self.fallback_responses[self._fallback_i % len(self.fallback_responses)]
        self._fallback_i += 1
        return response

    # ------------------------------------------------------------------
    def chat(self):
        print("Customer Service Bot: Hi! Ask me about orders, refunds, shipping, or billing.")
        print("(type 'quit' to exit)\n")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("Customer Service Bot: Goodbye!")
                break
            response = self.get_response(user_input)
            print(f"Customer Service Bot: {response}\n")


if __name__ == "__main__":
    bot = CustomerServiceChatbot(intents_path="intents.json")
    bot.chat()
