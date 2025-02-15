from config import DIRECTORY

import pixeltable as pxt

finance_agent = pxt.get_table(f'{DIRECTORY}.finance')

finance_agent.insert(prompt="What's the stock price of Nvidia?")
print(finance_agent.select(finance_agent.answer).collect())
