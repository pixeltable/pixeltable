import config
import pixeltable as pxt

from pixel.agent import create_agent
from pixel.index import create_index

# Create project
pxt.create_dir(config.PROJECT_NAME, if_exists='ignore')

# Create video index
video_table, video_index = create_index(
    index_name=config.VIDEO_INDEX_NAME,
    view_name=config.VIDEO_CHUNKS_NAME,
    purge=config.DELETE_ALL
)

# Insert sample video
videos = [
    config.VIDEO_FILE +
    f'Lex-Fridman-Podcast-430-Excerpt-{n}.mp4'
    for n in range(3)
]
video_table.insert({'video_file': video} for video in videos)

# Create agent
video_rag_agent = create_agent(
    agent_name=config.AGENT_NAME,
    index=video_index,
    system_prompt=config.SYSTEM_PROMPT,
    purge=config.DELETE_ALL
)

# Ask question
question = 'What is happiness?'
video_rag_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', video_rag_agent.answer.show())
