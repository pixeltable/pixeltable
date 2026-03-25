import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status
from .tool_utils import run_tool_invocations_test


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestMinimax:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('minimax')
        from pixeltable.functions.minimax import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(model='MiniMax-M2.5', messages=t.input_msgs))
        # test with additional parameters
        t.add_computed_column(
            chat_output_2=chat_completions(
                model='MiniMax-M2.5',
                messages=msgs,
                model_kwargs={
                    'max_tokens': 500,
                    'temperature': 0.7,
                    'top_p': 0.8,
                },
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['chat_output_2'][0]['choices'][0]['message']['content'].lower()

    def test_chat_completions_m27(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('minimax')
        from pixeltable.functions.minimax import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(chat_output=chat_completions(model='MiniMax-M2.7', messages=msgs))

        validate_update_status(t.insert(input='What is 2 + 2?'), 1)
        result = t.collect()
        assert '4' in result['chat_output'][0]['choices'][0]['message']['content']

    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('minimax')
        from pixeltable.functions import minimax

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=minimax.chat_completions(
                    model='MiniMax-M2.5', messages=messages, tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=minimax.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_random_question=False)
