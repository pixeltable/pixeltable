# pip install pixeltable replicate
import pixeltable as pxt
from pixeltable.functions import replicate

# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

# Create table
conversations = pxt.create_table(
    path_str="chatbot.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)

# Create the messages
conversations.add_computed_column(
    messages=[{"role": "user", "content": conversations.prompt}]
)

# Call OpenAI
conversations.add_computed_column(
    response=replicate.run(
        messages=conversations.messages,
        model="meta/meta-llama-3-8b-instruct",
    )
)

# Extract the answer
conversations.add_computed_column(
    answer=conversations.response.choices[0].message.content
)

# # Lets test it!!
conversations.insert([{"prompt": "Tell me a joke."}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())
