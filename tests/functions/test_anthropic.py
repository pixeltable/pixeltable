import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status
from .tool_utils import run_tool_invocations_test


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
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

    @rerun(reruns=6, reruns_delay=8)
    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions import anthropic

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
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
