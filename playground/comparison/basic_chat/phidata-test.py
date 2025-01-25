# pip install phidata
from typing import Optional
import json
import time

from phi.agent import Agent
from phi.model.anthropic import Claude

finance_agent = Agent(name='Comedian', model=Claude(id='claude-3-5-sonnet-20240620'))

start_time = time.time()
print('\n' + '=' * 50)
print('Processing query about NVDA price')
print('=' * 50 + '\n')

finance_agent.print_response('tell me a 4 word joke', stream=True)

end_time = time.time()
print('\n' + '-' * 50)
print(f'Time taken: {end_time - start_time:.2f} seconds')
print('-' * 50 + '\n')
