from config import DIRECTORY

import pixeltable as pxt

# Get tables
website_table = pxt.get_table(f'{DIRECTORY}.websites')
agent_table = pxt.get_table(f'{DIRECTORY}.conversations')

# Insert sample website
website_table.insert([{'website': 'https://quotes.toscrape.com/'}])

# Ask question
question = 'Explain the Albert Einstein quote'
agent_table.insert([{'prompt': question}])

# Show results
print('\nAnswer:', agent_table.answer.show())
