import asyncio
import logging
import subprocess
import time
from typing import Iterator

import pytest

import pixeltable as pxt

from .utils import skip_test_if_no_client, skip_test_if_not_installed

_logger = logging.getLogger('pixeltable')


class TestMcp:
    def test_mcp_server(self, reset_db: None, init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp')

        udfs = asyncio.run(pxt.mcp_udfs('http://localhost:8000/mcp'))
        assert udfs[0].name == 'pixelmultiple'
        assert udfs[0].comment() == 'Computes the Pixelmultiple of two integers.'
        assert udfs[1].name == 'pixeldict'
        assert udfs[1].comment() == 'Returns the Pixeldict of a dictionary.'

        t = pxt.create_table('test_mcp', {'a': pxt.Int, 'b': pxt.Int})
        t.add_computed_column(pixelmultiple=udfs[0](a=t.a, b=t.b))
        t.insert([{'a': 3, 'b': 4}, {'a': 5, 'b': 6}])
        res = t.select(t.pixelmultiple).head()
        assert res[0]['pixelmultiple'] == str((3 + 22) * 4)
        assert res[1]['pixelmultiple'] == str((5 + 22) * 6)

    def test_mcp_as_tools(self, reset_db: None, init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp')
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openai')
        from pixeltable.functions import openai

        udfs = asyncio.run(pxt.mcp_udfs('http://localhost:8000/mcp'))
        tools = pxt.tools(*udfs)

        t = pxt.create_table('test_mcp', {'prompt': pxt.String})
        messages = [{'role': 'user', 'content': t.prompt}]
        t.add_computed_column(response=openai.chat_completions(messages, model='gpt-4o-mini', tools=tools))
        t.add_computed_column(tool_calls=openai.invoke_tools(tools, t.response))
        t.insert(prompt='What is the pixelmultiple of 7 and 9?')
        res = t.head()
        assert res[0]['tool_calls'] == {'pixelmultiple': [str((7 + 22) * 9)], 'pixeldict': None}


@pytest.fixture(scope='session')
def init_mcp_server(init_env: None) -> Iterator[None]:
    skip_test_if_not_installed('mcp')

    _logger.info('Starting MCP server pytest fixture.')
    mcp_process = subprocess.Popen(['python', 'tests/example_mcp_server.py'])
    time.sleep(5)  # Wait for the MCP server to start
    yield

    _logger.info('Terminating MCP server pytest fixture.')
    mcp_process.kill()
