from pydantic_ai import Agent
import time

# Start timing
start_time = time.time()

agent = Agent('claude-3-5-sonnet-20240620')

result = agent.run_sync('Tell me a 4 word joke')
print(result.data)

# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f'\nTotal execution time: {elapsed_time:.2f} seconds')
