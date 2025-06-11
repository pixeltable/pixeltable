import pytest

import pixeltable as pxt

from .tool_utils import run_tool_invocations_test
from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestGroq:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('groq')
        skip_test_if_no_client('groq')
        from pixeltable.functions.groq import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=chat_completions(messages=msgs, model='llama3-8b-8192'))
        t.add_computed_column(
            output2=chat_completions(
                messages=msgs,
                model='llama3-8b-8192',
                model_kwargs={
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'max_tokens': 300,
                    'stop': ['\n'],
                    'response_format': {'type': 'text'},
                },
            )
        )
        validate_update_status(t.insert(input='What are the three most recently discovered chemical elements?'), 1)
        results = t.collect()
        assert 'tennessine' in results['output'][0]['choices'][0]['message']['content'].lower()
        assert len(results['output2'][0]['choices'][0]['message']['content']) > 0

    @pytest.mark.flaky(reruns=6, reruns_delay=8, condition=DO_RERUN)
    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openai')
        from pixeltable.functions import groq

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=groq.chat_completions(
                    model='llama3-8b-8192', messages=messages, tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=groq.invoke_tools(tools, t.response))
            return t

        # Tool choice is implemented, but its behavior in Groq differs subtly from other LLM providers, so there
        # are no tests currently (until we better understand Groq's behavior).
        # Groq also seems to do something random in response to a random question (unlike OpenAI, which consistently
        # calls no tools, as expected).
        run_tool_invocations_test(make_table, test_random_question=False)
