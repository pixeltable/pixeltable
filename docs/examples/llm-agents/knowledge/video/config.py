# Project params
PROJECT_NAME = 'video_agent'

# Index params
VIDEO_INDEX_NAME = f'{PROJECT_NAME}.video_index'
VIDEO_CHUNKS_NAME = f'{PROJECT_NAME}.video_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
VIDEO_FILE = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/'

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = True
# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the video file.

Use your search tool to find the most relevant information in the video file.

"""