# Customer Service Chatbot

A hybrid rule-based + simple-NLP chatbot that answers basic customer
support questions (orders, refunds, shipping, hours, billing, complaints).

## How it works
1. **Rule-based layer** — checks for exact keyword/phrase matches
   against patterns defined in `intents.json`.
2. **Simple NLP layer (fallback)** — if no exact match, the message is
   compared to every known pattern using **TF-IDF vectorization +
   cosine similarity**, so rephrased questions ("when's my stuff
   coming?") still map to the right intent ("where is my order").
3. **Fallback** — if similarity is too low, the bot admits it doesn't
   understand and asks the user to rephrase.

## Files
- `chatbot.py` — the chatbot class + interactive CLI loop
- `intents.json` — the knowledge base (tags, patterns, responses) —
  edit this to add new topics without touching any code
- `test_chatbot.py` — scripted demo, no typing needed

## Run it
```bash
pip install scikit-learn
python chatbot.py        # interactive chat in your terminal
python test_chatbot.py   # scripted demo with sample messages
```

## Adding a new topic
Just add a new block to `intents.json`:
```json
{
  "tag": "account_password",
  "patterns": ["reset my password", "forgot password", "cant log in"],
  "responses": ["I can help reset your password — what's the email on your account?"]
}
```
No code changes needed.

## Good talking points for your writeup
- Explains the difference between rule-based matching (fast, precise,
  brittle to rephrasing) and NLP similarity matching (more flexible,
  needs a confidence threshold to avoid wrong guesses).
- The `similarity_threshold` (default 0.35) is the key tuning knob —
  lower it to catch more rephrasings at the risk of more false
  matches; raise it to be more conservative.
- Natural extensions: add spaCy/NLTK for lemmatization so "orders"
  and "order" always match; log unmatched questions to see what real
  users ask and expand `intents.json` over time; swap in a small
  transformer-based intent classifier once you have labeled data.
