# Project params
PROJECT_NAME = 'team_of_agents'

# Index params

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = False

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.leader_agent'
SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the audio file.

Use your search tool to find the most relevant information in the audio file.

"""