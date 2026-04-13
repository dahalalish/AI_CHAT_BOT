import random

SMALL_TALK = [
    "hello", "hi", "hey", "thanks", "thank you",
    "good morning", "good evening"
]

TOXIC_WORDS = [
    "idiot", "stupid", "dumb", "useless", "trash"
]

CLEAR_COMMANDS = ["clear", "reset", "start over"]


def detect_intent(query: str):

    q = query.lower().strip()

    if q in CLEAR_COMMANDS:
        return "CLEAR"

    if any(word in q for word in SMALL_TALK):
        return "SMALL_TALK"

    if any(word in q for word in TOXIC_WORDS):
        return "TOXIC"

    return "NORMAL"


def generate_smalltalk_response():
    responses = [
        "Hey there 👋 Ready to explore payer data?",
        "Hello! Ask me anything about payer mappings.",
        "Hi! I specialize in payer datasets — try me 😄"
    ]
    return random.choice(responses)


def generate_toxic_response():
    responses = [
        "😄 I’ll ignore that and help you with payer data instead.",
        "Let’s keep it professional — ask me about payers!",
        "I may be an AI, but I prefer data over drama 😎"
    ]
    return random.choice(responses)


def generate_out_of_scope_response():
    return """
I’m designed to help with payer-related queries.

Try asking:
• Which payers use member_id?
• Show mapping for Aetna
• Explain business logic for plan_id
"""
