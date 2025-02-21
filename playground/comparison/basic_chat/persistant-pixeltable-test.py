import pixeltable as pxt
import time

# Create table
conversations = pxt.get_table('chatbot.conversations')

# Start timing
start_time = time.time()

# # Lets test it!!
conversations.insert([{'prompt': 'Tell me a 4 word joke.'}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f'\nTotal execution time: {elapsed_time:.2f} seconds')
