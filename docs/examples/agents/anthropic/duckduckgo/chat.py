from config import DIRECTORY

import pixeltable as pxt

news_agent = pxt.get_table(f'{DIRECTORY}.news')

news_agent.insert(prompt="What's the latest news in Los Angeles?")
print(news_agent.select(news_agent.tool_output, news_agent.answer).collect())
