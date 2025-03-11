from typing import Optional

import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, stock_price, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8)
class TestAnthropic:
    def test_anthropic(self, reset_db) -> None:
        from pixeltable.functions.anthropic import messages

        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        t = pxt.create_table('test_tbl', {'input': pxt.String})

        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=messages(messages=msgs, model='claude-3-haiku-20240307'))
        t.add_computed_column(
            output2=messages(
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
        )
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['content'][0]['text']) > 0
        assert len(results['output2'][0]['content'][0]['text']) > 0

    @pytest.mark.flaky(reruns=6, reruns_delay=8)
    def test_tool_invocations(self, reset_db) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions.anthropic import invoke_tools, messages

        # stock_price is a module UDF and weather is a local UDF, so we test both
        @pxt.udf(_force_stored=True)
        def weather(city: str) -> Optional[str]:
            """
            Get today's weather forecast for a given city.

            Args:
                city - The name of the city to look up.
            """
            if city == 'San Francisco':
                return 'Cloudy with a chance of meatballs'
            else:
                return 'Unknown city'

        tools = pxt.tools(stock_price, weather)
        tool_choice_opts: list[Optional[pxt.func.ToolChoice]] = [
            None,
            tools.choice(auto=True),
            tools.choice(required=True),
            tools.choice(tool='stock_price'),
            tools.choice(tool=weather),
            tools.choice(required=True, parallel_tool_calls=False),
        ]

        for tool_choice in tool_choice_opts:
            pxt.drop_table('test_tbl', if_not_exists='ignore')
            t = pxt.create_table('test_tbl', {'prompt': pxt.String})
            msgs = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=messages(
                    model='claude-3-5-sonnet-20241022', messages=msgs, tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
            t.insert(prompt='What is the stock price of NVDA today?')
            t.insert(prompt='What is the weather in San Francisco?')
            t.insert(prompt='What is the stock price of NVDA today, and what is the weather in San Francisco?')
            t.insert(prompt='How many grams of corn are in a bushel?')
            t.insert(prompt='What is the stock price of NVDA today? Also, what is the stock price of UAL?')
            res = t.select(t.response, t.tool_calls).head()
            print(f'Responses with tool_choice equal to: {tool_choice}')
            print(res[0]['response'])
            print(res[1]['response'])
            print(res[2]['response'])
            print(res[3]['response'])
            print(res[4]['response'])

            # Request for stock price: works except when tool_choice is set explicitly to weather
            print('Checking stock price inquiry')
            if tool_choice is None or tool_choice.tool != 'weather':
                assert res[0]['tool_calls'] == {'stock_price': [131.17], 'weather': None}
            else:  # Explicitly set to weather; we may or may not get stock price also
                assert res[0]['tool_calls'] in [
                    {'stock_price': None, 'weather': ['Unknown city']},
                    {'stock_price': [131.17], 'weather': ['Unknown city']},
                ]

            # Request for weather: works except when tool_choice is set explicitly to stock_price
            print('Checking weather inquiry')
            if tool_choice is None or tool_choice.tool != 'stock_price':
                assert res[1]['tool_calls'] == {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']}
            else:  # Explicitly set to stock_price; we may or may not get weather also
                assert res[1]['tool_calls'] in [
                    {'stock_price': [0.0], 'weather': None},
                    {'stock_price': [0.0], 'weather': ['Cloudy with a chance of meatballs']},
                ]

            # Request for both stock price and weather
            print('Checking double inquiry')
            if tool_choice is None or tool_choice.parallel_tool_calls:
                # Both tools invoked in parallel
                assert res[2]['tool_calls'] == {
                    'stock_price': [131.17],
                    'weather': ['Cloudy with a chance of meatballs'],
                }
            else:
                # Only one tool invoked, but it's not specified which
                assert not tool_choice.parallel_tool_calls
                assert res[2]['tool_calls'] in [
                    {'stock_price': [131.17], 'weather': None},
                    {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']},
                ]

            print('Checking random question')
            if tool_choice is None or tool_choice.auto:
                assert res[3]['tool_calls'] == {'stock_price': None, 'weather': None}
            elif tool_choice.tool == 'stock_price':
                assert res[3]['tool_calls'] == {'stock_price': [0.0], 'weather': None}
            elif tool_choice.tool == 'weather':
                assert res[3]['tool_calls'] == {'stock_price': None, 'weather': ['Unknown city']}
            else:
                assert res[3]['tool_calls'] in [
                    {'stock_price': [0.0], 'weather': None},
                    {'stock_price': None, 'weather': ['Unknown city']},
                ]

            print('Checking multiple stock prices question')
            if tool_choice is None or tool_choice.auto:
                assert res[4]['tool_calls'] == {'stock_price': [131.17, 82.88], 'weather': None}
