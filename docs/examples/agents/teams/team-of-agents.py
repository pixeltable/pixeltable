import config
import pixeltable as pxt

from pixel.agent import create_agent

import yfinance as yf

# Create project
pxt.create_dir(config.PROJECT_NAME, if_exists='ignore')

# Create agent
audio_rag_agent = create_agent(
    agent_name="financial_research_agent",
    agent_tools=tools,
    system_prompt=config.SYSTEM_PROMPT,
    purge=config.DELETE_ALL
)

# Ask question
question = 'What is Pixeltable?'
audio_rag_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', audio_rag_agent.answer.show())
