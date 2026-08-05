from chatbot import CustomerServiceChatbot

bot = CustomerServiceChatbot(intents_path="intents.json")

test_messages = [
    "hi there",
    "where is my order",
    "when's my stuff coming",       # rephrased -> tests NLP fuzzy match
    "I want to return this, how do refunds work",
    "how much does shipping cost",
    "what time do you open",
    "your product broke and I'm furious",
    "can I pay with paypal",
    "asdkjhaskjdh random gibberish",  # should trigger fallback
    "thanks so much",
    "bye",
]

for msg in test_messages:
    print(f"You: {msg}")
    print(f"Bot: {bot.get_response(msg)}\n")
