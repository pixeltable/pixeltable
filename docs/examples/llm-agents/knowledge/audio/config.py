# Project params
PROJECT_NAME = 'audio_agent'

# Index params
AUDIO_INDEX_NAME = f'{PROJECT_NAME}.audio_index'
AUDIO_CHUNKS_NAME = f'{PROJECT_NAME}.audio_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
AUDIO_FILE = 's3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3'

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = False

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the audio file.

Use your search tool to find the most relevant information in the audio file.

"""