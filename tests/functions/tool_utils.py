from typing import Callable, Optional

import pixeltable as pxt


def run_tool_invocations_test(
    make_table: Callable[[pxt.func.Tools, pxt.func.ToolChoice], pxt.Table],
    *,
    test_random_question: bool = True,
    test_multiple_tool_use: bool = True,
    test_tool_choice: bool = False,
    test_individual_tool_choice: bool = False,
) -> None:
    """make_table is expected to yield an empty table with 'prompt' and 'tool_calls' columns."""
    tools = pxt.tools(stock_price, weather)
    tool_choice_opts: list[Optional[pxt.func.ToolChoice]] = [None]
    if test_tool_choice:
        tool_choice_opts += [
            tools.choice(auto=True),
            tools.choice(required=True),
            tools.choice(required=True, parallel_tool_calls=False),
        ]
    if test_individual_tool_choice:
        tool_choice_opts += [
            tools.choice(tool='stock_price'),  # Specified by name
            tools.choice(tool=weather),  # Specified by function
        ]

    for tool_choice in tool_choice_opts:
        t = make_table(tools, tool_choice)
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
        print(f'Checking stock price inquiry [tool_choice: {tool_choice}]')
        if tool_choice is None or tool_choice.tool != 'weather':
            assert res[0]['tool_calls'] == {'stock_price': [131.17], 'weather': None}
        else:  # Explicitly set to weather; we may or may not get stock price also
            assert res[0]['tool_calls'] in [
                {'stock_price': None, 'weather': ['Unknown city']},
                {'stock_price': [131.17], 'weather': ['Unknown city']},
            ]

        # Request for weather: works except when tool_choice is set explicitly to stock_price
        print(f'Checking weather inquiry [tool_choice: {tool_choice}]')
        if tool_choice is None or tool_choice.tool != 'stock_price':
            assert res[1]['tool_calls'] == {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']}
        else:  # Explicitly set to stock_price; we may or may not get weather also
            assert res[1]['tool_calls'] in [
                {'stock_price': [0.0], 'weather': None},
                {'stock_price': [0.0], 'weather': ['Cloudy with a chance of meatballs']},
            ]

        if test_random_question:
            print(f'Checking random question [tool_choice: {tool_choice}]')
            if tool_choice is None or tool_choice.auto:
                assert res[3]['tool_calls'] == {'stock_price': None, 'weather': None}, res[3]['tool_calls']
            elif tool_choice.tool == 'stock_price':
                assert res[3]['tool_calls'] == {'stock_price': [0.0], 'weather': None}
            elif tool_choice.tool == 'weather':
                assert res[3]['tool_calls'] == {'stock_price': None, 'weather': ['Unknown city']}
            else:
                assert res[3]['tool_calls'] in [
                    {'stock_price': [0.0], 'weather': None},
                    {'stock_price': None, 'weather': ['Unknown city']},
                ]

        if test_multiple_tool_use:
            # Request for both stock price and weather
            print(f'Checking double inquiry [tool_choice: {tool_choice}]')
            if tool_choice is None or (tool_choice.parallel_tool_calls and tool_choice.tool is None):
                # Both tools invoked in parallel
                assert res[2]['tool_calls'] == {
                    'stock_price': [131.17],
                    'weather': ['Cloudy with a chance of meatballs'],
                }
            elif tool_choice.tool == 'stock_price':
                assert res[2]['tool_calls'] == {'stock_price': [131.17], 'weather': None}
            elif tool_choice.tool == 'weather':
                assert res[2]['tool_calls'] == {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']}
            else:
                # Only one tool invoked, but it's not specified which
                assert not tool_choice.parallel_tool_calls
                assert res[2]['tool_calls'] in [
                    {'stock_price': [131.17], 'weather': None},
                    {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']},
                ]

            print(f'Checking multiple stock prices question [tool_choice: {tool_choice}]')
            if tool_choice is None or tool_choice.auto:
                # If you specify an explicit tool, it seems to only call it once.
                assert res[4]['tool_calls'] == {'stock_price': [131.17, 82.88], 'weather': None}


# Mock UDF for testing LLM tool invocations
@pxt.udf
def stock_price(ticker: str) -> float:
    """
    Get today's stock price for a given ticker symbol.

    Args:
        ticker - The ticker symbol of the stock to look up.
    """
    if ticker == 'NVDA':
        return 131.17
    elif ticker == 'UAL':
        return 82.88
    else:
        # Return 0.0 instead of None, to distinguish between these two cases: the tool not being called, and the tool
        # being called on a symbol other than NVDA/UAL
        return 0.0


@pxt.udf
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


@pxt.udf
def server_state() -> str:
    """
    Get the current server state.

    Returns:
        The current server state.
    """
    return 'Running (0x4171780)'
