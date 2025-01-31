from pxt.agent import Agent
from pxt.providers import Model

# Create Agent
llm = Model(provider='openai', model_name='gpt-4o-mini', temperature=0.1)
agent = Agent(
    model=llm,
    agent_name='Dog Trainer',
    system_prompt='You specialize in golden retriever training.',
    # clear_cache=True, # Remove for persistent agent
)

# Get answer
result = agent.run('Tell me how to train a golden retriever to sit in 10 words.')
print(result)

# Inspect agent history
inspect = agent.get_history()
df = inspect.collect().to_pandas()
print(df.head())
print(df.columns)
