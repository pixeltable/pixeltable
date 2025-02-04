from config import DIRECTORY

import pixeltable as pxt

# Load the table
conversations = pxt.get_table(f'{DIRECTORY}.conversations')

# # Lets test it!!
conversations.insert([{'prompt': 'Tell me a 4 word joke.'}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())
