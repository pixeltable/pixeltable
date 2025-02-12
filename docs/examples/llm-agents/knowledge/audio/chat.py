import config
import pixeltable as pxt

from pixel.agent import create_agent
from pixel.index import create_index

# Create project
pxt.create_dir(config.PROJECT_NAME, if_exists='ignore')

# Create audio index
audio_table, audio_index = create_index(
    index_name=config.AUDIO_INDEX_NAME,
    chunks_name=config.AUDIO_CHUNKS_NAME,
    purge=config.DELETE_ALL
)

# Insert sample audio
audio_table.insert([{'audio_file': config.AUDIO_FILE}])

# Create agent
audio_rag_agent = create_agent(
    agent_name=config.AGENT_NAME,
    index=audio_index,
    system_prompt=config.SYSTEM_PROMPT,
    purge=config.DELETE_ALL
)

# Ask question
question = 'What is Pixeltable?'
audio_rag_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', audio_rag_agent.answer.show())
