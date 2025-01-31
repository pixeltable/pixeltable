from pxt.agent import Agent

import uuid

# Create agent
agent_uuid = str(uuid.uuid4())
agent = Agent(
    agent_name     = f"audio_agent_{agent_uuid}", 
    system_prompt  = "You are a helpful assistant.", 
    reset_session  = False
)

# Get answer
result = agent.run("Write a 4 word haiku about dogs.")

print(result)

