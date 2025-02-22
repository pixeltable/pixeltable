import pixeltable as pxt
from pixeltable.functions import anthropic
import time

# Start timing
start_time = time.time()

# Initialize Pixeltable
pxt.drop_dir('chatbot', force=True)
pxt.create_dir('chatbot')

# Create table
conversations = pxt.create_table(
    path_str='chatbot.conversations',
    schema_or_df={'prompt': pxt.String},
    if_exists='ignore',
)

# Create the messages
conversations.add_computed_column(messages=[{'role': 'user', 'content': conversations.prompt}])

# Call Anthropic
conversations.add_computed_column(
    response=anthropic.messages(
        messages=conversations.messages,
        model='claude-3-5-sonnet-20240620',
    )
)

# Extract the answer
conversations.add_computed_column(answer=conversations.response.content)






# # Lets test it!!
conversations.insert([{'prompt': 'Tell me a joke.'}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f'\nTotal execution time: {elapsed_time:.2f} seconds')
