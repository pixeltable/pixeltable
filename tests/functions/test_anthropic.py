import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, stock_price, validate_update_status


@pytest.mark.remote_api
class TestAnthropic:
    def test_anthropic(self, reset_db) -> None:
        from pixeltable.functions.anthropic import messages

        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        t = pxt.create_table('test_tbl', {'input': pxt.String})

        msgs = [{'role': 'user', 'content': t.input}]
        t['output'] = messages(messages=msgs, model='claude-3-haiku-20240307')
        t['output2'] = messages(
            messages=msgs,
            model='claude-3-haiku-20240307',
            max_tokens=300,
            metadata={'user_id': 'pixeltable'},
            stop_sequences=['STOP'],
            system='You are an ordinary person walking down the street.',
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['content'][0]['text']) > 0
        assert len(results['output2'][0]['content'][0]['text']) > 0

    def test_tool_invocations(self, reset_db) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions.anthropic import invoke_tools, messages

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        msgs = [{'role': 'user', 'content': t.prompt}]
        tools = pxt.tools(stock_price)
        t.add_computed_column(response=messages(
            model='claude-3-haiku-20240307',
            messages=msgs,
            tools=tools
        ))
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
        t.insert(prompt='What is the stock price of NVDA today?')
        t.insert(prompt='How many grams of corn are in a bushel?')
        res = t.select(t.response, t.tool_calls).head()

        # First prompt results in tool invocation
        # (with Anthropic, there may also be a text response such as 'Ok, let me look up the stock price.')
        assert res[0]['tool_calls'] == {'stock_price': 131.17}

        # Second prompt results in no tool invocation
        assert res[1]['tool_calls'] == {'stock_price': None}
