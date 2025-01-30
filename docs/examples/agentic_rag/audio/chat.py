from config import AUDIO_FILE, DIRECTORY

import pixeltable as pxt

# Get tables
audio_table = pxt.get_table(f'{DIRECTORY}.audio')
agent_table = pxt.get_table(f'{DIRECTORY}.conversations')

# Insert sample audio
audio_table.insert([{'audio_file': AUDIO_FILE}])

# Ask question
question = 'What is pixeltable?'
agent_table.insert([{'prompt': question}])

# Show results
print('\nAnswer:', agent_table.answer.show())
