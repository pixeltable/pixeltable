import pixeltable as pxt
from datetime import datetime

# Fetch table
conversations = pxt.get_table("chatbot")

# Insert data
conversations.insert(
    [{"prompt": "What was my last question?", "timestamp": datetime.now()}]
)

# Print the table
print(conversations.select(conversations.memory_answer).collect())
