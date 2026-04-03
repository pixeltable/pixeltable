"""UDFs for perftest_providers.py. Separated into a module to satisfy Pixeltable's UDF requirements."""

import pixeltable as pxt


def instructions(count: int, word1: str, word2: str) -> str:
    return (
        f'Generate a roughly {count}-token essay that uses both of these words: {word1} and {word2}. '
        f'The prose should be natural and make sense.'
    )


@pxt.udf
def create_chatgpt_prompt(count: int, word1: str, word2: str) -> list[dict[str, str]]:
    """Create a prompt in ChatGPT message format."""
    return [
        {'role': 'system', 'content': 'You are a creative writer who creates natural-sounding prose.'},
        {'role': 'user', 'content': instructions(count, word1, word2)},
    ]


@pxt.udf
def create_simple_messages_prompt(count: int, word1: str, word2: str) -> list[dict[str, str]]:
    """Create a prompt in Claude message format (no system message in messages)."""
    return [{'role': 'user', 'content': instructions(count, word1, word2)}]


@pxt.udf
def create_simple_prompt(count: int, word1: str, word2: str) -> str:
    """Create a simple string prompt."""
    return f'You are a creative writer who creates natural-sounding prose. {instructions(count, word1, word2)}'
