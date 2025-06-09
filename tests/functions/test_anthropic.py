from typing import Optional

import pytest

import pixeltable as pxt
from tests.functions.tool_invocations import run_tool_invocations_test

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, stock_price, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestAnthropic:
    def test_messages(self, reset_db: None) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions import anthropic

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=anthropic.messages(messages=messages, model='claude-3-haiku-20240307', max_tokens=1024)
        )
        t.add_computed_column(
            output2=anthropic.messages(
                messages=messages,
                model='claude-3-haiku-20240307',
                max_tokens=300,
                model_kwargs={
                    'metadata': {'user_id': 'pixeltable'},
                    'stop_sequences': ['STOP'],
                    'system': 'You are an ordinary person walking down the street.',
                    'temperature': 0.7,
                    'top_k': 40,
                    'top_p': 0.9,
                },
            )
        )
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['content'][0]['text']) > 0
        assert len(results['output2'][0]['content'][0]['text']) > 0

    @pytest.mark.flaky(reruns=6, reruns_delay=8, condition=DO_RERUN)
    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions import anthropic

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=anthropic.messages(
                    model='claude-3-5-sonnet-20241022',
                    messages=messages,
                    max_tokens=1024,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )
            t.add_computed_column(tool_calls=anthropic.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_tool_choice=True)
