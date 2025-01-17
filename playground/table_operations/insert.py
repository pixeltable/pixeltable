import pixeltable as pxt
from datetime import datetime

# Fetch table
conversations = pxt.get_table("chatbot")

# Insert data
conversations.insert(
    [{"prompt": "Write a haiku about saint bernards.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about dolphines.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about cats.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about dogs.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about horses.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about hereford cows.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about angus cows.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about sheep.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about goats.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about pigs.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about chickens.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about ducks.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about geese.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about turkeys.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "Write a haiku about berkshire pigs.", "timestamp": datetime.now()}]
)
conversations.insert(
    [{"prompt": "What was my last question?", "timestamp": datetime.now()}]
)
