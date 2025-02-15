import pixeltable as pxt

from index import create_index
from agent import create_agent

# Project params
PROJECT_NAME = 'video_agent'

# Index params
VIDEO_INDEX_NAME = f'{PROJECT_NAME}.video_index'
VIDEO_CHUNKS_NAME = f'{PROJECT_NAME}.video_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
VIDEO_FILE = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/'

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = "You are a helpful assistant that can answer questions about the video file."

# Create project
pxt.create_dir(PROJECT_NAME, if_exists='ignore')

# Create video base table and index
create_index(
    index_name=VIDEO_INDEX_NAME,
    view_name=VIDEO_CHUNKS_NAME,
    reset_history=False
)

# The base table holds metadata about the video file
video_table = pxt.get_table(VIDEO_INDEX_NAME)

# Insert sample video
videos = [
    VIDEO_FILE +
    f'Lex-Fridman-Podcast-430-Excerpt-{n}.mp4'
    for n in range(3)
]
video_table.insert({'video_file': video} for video in videos)

# The index holds the embeddings and the chunked text to retrieve
video_index = pxt.get_table(VIDEO_INDEX_NAME)

# Create agent
create_agent(
    agent_name=AGENT_NAME,
    index=video_index,
    llm_model_name=AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    reset_history=False
)

# Ask question
video_agent = pxt.get_table(AGENT_NAME)
question = 'What is happiness?'
video_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', video_agent.answer.show())
