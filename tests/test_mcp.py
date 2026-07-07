import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

import pixeltable as pxt

from .utils import reload_catalog, rerun, skip_test_if_no_client, skip_test_if_not_installed

_logger = logging.getLogger('pixeltable_test')


@rerun(reruns=3, delay=30)
class TestMcp:
    def test_mcp_server(self, make_catalog_path: Callable[[str], str], init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp')

        udfs = pxt.mcp_udfs('http://localhost:8000/mcp')
        assert udfs[0].name == 'pixelmultiple'
        assert udfs[0].comment() == 'Computes the Pixelmultiple of two integers.'
        assert udfs[1].name == 'pixeldict'
        assert udfs[1].comment() == 'Returns the Pixeldict of a dictionary.'

        t = pxt.create_table(make_catalog_path('test_mcp'), {'a': pxt.Int, 'b': pxt.Int})
        t.add_computed_column(pixelmultiple=udfs[0](a=t.a, b=t.b))
        t.insert([{'a': 3, 'b': 4}, {'a': 5, 'b': 6}])
        res = t.order_by(t.a).collect()
        assert res[0]['pixelmultiple'] == str((3 + 22) * 4)
        assert res[1]['pixelmultiple'] == str((5 + 22) * 6)

    def test_mcp_persistence(self, make_catalog_path: Callable[[str], str], init_mcp_server: None) -> None:
        # a column backed by an MCP UDF survives a catalog reload: its stored values read back, and the
        # reconstructed function still computes newly inserted rows against the server
        skip_test_if_not_installed('mcp')

        path = make_catalog_path('test_mcp')
        udfs = pxt.mcp_udfs('http://localhost:8000/mcp')
        t = pxt.create_table(path, {'a': pxt.Int, 'b': pxt.Int})
        t.add_computed_column(pixelmultiple=udfs[0](a=t.a, b=t.b))
        t.insert([{'a': 3, 'b': 4}])

        reload_catalog()
        t = pxt.get_table(path)
        assert t.where(t.a == 3).collect()[0]['pixelmultiple'] == str((3 + 22) * 4)
        t.insert([{'a': 5, 'b': 6}])
        assert t.where(t.a == 5).collect()[0]['pixelmultiple'] == str((5 + 22) * 6)

    def test_mcp_tool_changed(self, make_catalog_path: Callable[[str], str]) -> None:
        # evaluating an MCP-backed column reports a clear error if the server's tool set has drifted since the
        # column was created: the tool's signature changed, or the tool is gone entirely
        skip_test_if_not_installed('mcp')
        url = 'http://localhost:8001/mcp'

        # create the column against the (a, b) tool, but don't compute a row yet: the tool is checked lazily on
        # first evaluation, so leaving it unchecked here lets the drift below be detected when a row is inserted
        with _mcp_server_variant('full'):
            udfs = pxt.mcp_udfs(url)
            pixelmultiple = next(udf for udf in udfs if udf.name == 'pixelmultiple')
            t = pxt.create_table(make_catalog_path('test_mcp'), {'a': pxt.Int, 'b': pxt.Int})
            t.add_computed_column(pixelmultiple=pixelmultiple(a=t.a, b=t.b))

        # the live 'pixelmultiple' tool now takes only (a), no longer matching the stored (a, b) signature
        with _mcp_server_variant('changed'):
            assert t.insert([{'a': 5, 'b': 6}], on_error='ignore').num_excs > 0
            assert 'has changed' in t.where(t.a == 5).select(m=t.pixelmultiple.errormsg).collect()[0]['m']

        # the tool is no longer advertised at all
        with _mcp_server_variant('gone'):
            assert t.insert([{'a': 7, 'b': 8}], on_error='ignore').num_excs > 0
            assert 'no longer available' in t.where(t.a == 7).select(m=t.pixelmultiple.errormsg).collect()[0]['m']

    def test_mcp_as_tools(self, make_catalog_path: Callable[[str], str], init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp', 'openai')
        skip_test_if_no_client('openai')
        from pixeltable.functions import openai

        udfs = pxt.mcp_udfs('http://localhost:8000/mcp')
        tools = pxt.tools(*udfs)

        t = pxt.create_table(make_catalog_path('test_mcp'), {'prompt': pxt.String})
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
    mcp_process = subprocess.Popen([sys.executable, 'tests/example_mcp_server.py'])
    time.sleep(5)  # Wait for the MCP server to start
    yield

    _logger.info('Terminating MCP server pytest fixture.')
    mcp_process.kill()


@contextmanager
def _mcp_server_variant(variant: str) -> Iterator[None]:
    # Run the example server on a dedicated port so its lifecycle is independent of the session-scoped server.
    env = {**os.environ, 'PIXELTABLE_MCP_PORT': '8001', 'PIXELTABLE_MCP_VARIANT': variant}
    process = subprocess.Popen([sys.executable, 'tests/example_mcp_server.py'], env=env)
    time.sleep(5)  # Wait for the MCP server to start
    try:
        yield
    finally:
        process.kill()
        process.wait()
