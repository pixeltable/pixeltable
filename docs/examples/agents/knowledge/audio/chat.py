import pixeltable as pxt

from agent import create_agent
from index import create_index

# Project name
PROJECT_NAME = 'audio_agent'

# Index params
AUDIO_INDEX_NAME = f'{PROJECT_NAME}.audio_index'
AUDIO_CHUNKS_NAME = f'{PROJECT_NAME}.audio_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
AUDIO_FILE = 's3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3'

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the audio file.

Use your search tool to find the most relevant information in the audio file.
"""

# Create project
pxt.create_dir(PROJECT_NAME, if_exists='ignore')

# Create audio index
create_index(
    index_name=AUDIO_INDEX_NAME,
    chunks_name=AUDIO_CHUNKS_NAME,
    reset_history=False
)

# The base table holds metadata about the audio file
audio_table = pxt.get_table(AUDIO_INDEX_NAME)

# Insert sample audio
audio_table.insert([{'audio_file': AUDIO_FILE}])

# The index holds the embeddings of the audio file and the chunked text to retrieve
audio_index = pxt.get_table(AUDIO_INDEX_NAME)

# Create agent
create_agent(
    agent_name=AGENT_NAME,
    index=audio_index,
    llm_model_name=AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    reset_history=False
)

# Ask question
audio_rag_agent = pxt.get_table(AGENT_NAME)
question = 'What is Pixeltable?'
audio_rag_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', audio_rag_agent.answer.show())
