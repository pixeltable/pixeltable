from typing import Any, Callable

import pixeltable as pxt


def run_tool_invocations_test(
    make_table: Callable[[pxt.Tools, pxt.ToolChoice], pxt.Table],
    *,
    test_non_tool_question: bool = True,
    test_multiple_tool_use: bool = True,
    test_tool_choice: bool = False,
    test_individual_tool_choice: bool = False,
) -> None:
    """make_table is expected to yield an empty table with 'prompt' and 'tool_calls' columns."""
    tools = pxt.tools(stock_price, weather)
    tool_choice_opts: list[pxt.ToolChoice | None] = [None]
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
        print(f'Running tests with tool_choice: {tool_choice}')
        t = make_table(tools, tool_choice)

        def prompt(text: str) -> Any:
            return t.insert(prompt=text, return_rows=True).rows[0]['tool_calls']

        # Request for stock price: works except when tool_choice is set explicitly to weather
        nvda_stock_tool_calls = prompt('What is the stock price of NVDA today?')
        if tool_choice is None or tool_choice.tool != 'weather':
            assert nvda_stock_tool_calls == {'stock_price': [131.17], 'weather': None}, nvda_stock_tool_calls
        else:  # Explicitly set to weather; we may or may not get stock price also
            assert nvda_stock_tool_calls in [
                {'stock_price': None, 'weather': ['Unknown city']},
                {'stock_price': [131.17], 'weather': ['Unknown city']},
            ], nvda_stock_tool_calls

        # Request for weather: works except when tool_choice is set explicitly to stock_price
        sf_weather_tool_calls = prompt('What is the weather in San Francisco?')
        if tool_choice is None or tool_choice.tool != 'stock_price':
            assert sf_weather_tool_calls == {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']}, (
                sf_weather_tool_calls
            )
        else:  # Explicitly set to stock_price; we may or may not get weather also
            assert sf_weather_tool_calls in [
                {'stock_price': [0.0], 'weather': None},
                {'stock_price': [0.0], 'weather': ['Cloudy with a chance of meatballs']},
            ], sf_weather_tool_calls

        if test_non_tool_question:
            non_tool_tool_calls = prompt('How many grams of corn are in a bushel?')
            if tool_choice is None or tool_choice.auto:
                assert non_tool_tool_calls == {'stock_price': None, 'weather': None}, non_tool_tool_calls
            elif tool_choice.tool == 'stock_price':
                assert non_tool_tool_calls == {'stock_price': [0.0], 'weather': None}
            elif tool_choice.tool == 'weather':
                assert non_tool_tool_calls == {'stock_price': None, 'weather': ['Unknown city']}
            else:
                assert non_tool_tool_calls in [
                    {'stock_price': [0.0], 'weather': None},
                    {'stock_price': None, 'weather': ['Unknown city']},
                ], non_tool_tool_calls

        if test_multiple_tool_use:
            # Request for both stock price and weather
            double_inquiry_tool_calls = prompt(
                'What is the stock price of NVDA today, and what is the weather in San Francisco?'
            )
            if tool_choice is None or (tool_choice.parallel_tool_calls and tool_choice.tool is None):
                # Both tools invoked in parallel
                assert double_inquiry_tool_calls == {
                    'stock_price': [131.17],
                    'weather': ['Cloudy with a chance of meatballs'],
                }
            elif tool_choice.tool == 'stock_price':
                assert double_inquiry_tool_calls == {'stock_price': [131.17], 'weather': None}
            elif tool_choice.tool == 'weather':
                assert double_inquiry_tool_calls == {
                    'stock_price': None,
                    'weather': ['Cloudy with a chance of meatballs'],
                }
            else:
                # Only one tool invoked, but it's not specified which
                assert not tool_choice.parallel_tool_calls
                assert double_inquiry_tool_calls in [
                    {'stock_price': [131.17], 'weather': None},
                    {'stock_price': None, 'weather': ['Cloudy with a chance of meatballs']},
                ], double_inquiry_tool_calls

            if tool_choice is None or tool_choice.auto:
                multiple_stocks_tool_calls = prompt(
                    'What is the stock price of NVDA today? Also, what is the stock price of UAL?'
                )
                # If you specify an explicit tool, it seems to only call it once.
                assert multiple_stocks_tool_calls == {'stock_price': [131.17, 82.88], 'weather': None}


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
def weather(city: str) -> str | None:
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
