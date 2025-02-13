import pixeltable as pxt

# Fetch table
conversations = pxt.get_table("chatbot")

# Print the table
print(
    conversations.select(
        conversations.answer, conversations.prompt, conversations.timestamp
    ).collect()
)

# Print the table
print(conversations.select(conversations.messages).collect())
