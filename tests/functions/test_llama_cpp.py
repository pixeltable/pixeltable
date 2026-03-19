from collections.abc import Generator

import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_not_installed, validate_update_status
from .tool_utils import stock_price, weather


@pytest.fixture(autouse=True)
def cleanup_llama_cpp() -> Generator:
    yield
    try:
        from pixeltable.functions import llama_cpp

        llama_cpp.cleanup()
    except Exception as e:
        print(f'cleanup failed: {e}')


@rerun(reruns=3, reruns_delay=15)  # Since it involes a HF model download
class TestLlamaCpp:
    def test_create_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('llama_cpp')
        from pixeltable.functions import llama_cpp

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t.add_computed_column(
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': t.input},
            ]
        )

        # We use a small model (~350MB) for testing.
        t.add_computed_column(
            output=llama_cpp.create_chat_completion(
                t.messages, repo_id='Qwen/Qwen2-0.5B-Instruct-GGUF', repo_filename='*q3_k_m.gguf'
            )
        )
        t.add_computed_column(
            output2=llama_cpp.create_chat_completion(
                t.messages,
                repo_id='Qwen/Qwen2-0.5B-Instruct-GGUF',
                repo_filename='*q3_k_m.gguf',
                model_kwargs={'max_tokens': 100, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 20},
            )
        )

        validate_update_status(t.insert(input='What are some edible species of fish?'), expected_rows=1)
        result = t.collect()['output'][0]
        result2 = t.collect()['output2'][0]
        print(result)
        assert len(result['choices'][0]['message']['content']) > 0
        assert len(result2['choices'][0]['message']['content']) > 0

    @pytest.mark.parametrize('model', ['mistral', 'gemma', 'qwen', 'salesforce'])
    def test_tool_invocations_3(self, uses_db: None, model: str) -> None:
        skip_test_if_not_installed('llama_cpp')
        from pixeltable.functions import llama_cpp
        from pixeltable.functions.openai import invoke_tools

        tools = pxt.tools(stock_price, weather)

        match model:
            case 'qwen':
                repo_id = 'Qwen/Qwen3-0.6B-GGUF'
                repo_filename = '*Q8_0.gguf'
                # The 'chatml-function-calling' chat format results in an extremely verbose prompt from this model
                chat_format = None
                # 'auto' tool choice causes Qwen to generate the tool call in its own native XML format, so specify
                # the tool choice explicitly instead.
                # If this limitation is unacceptable, we can implement a special case in the UDF that parses Qwen's
                # XML output to a dict.
                tool_choice = tools.choice(tool=weather)
            case 'salesforce':
                repo_id = 'Salesforce/Llama-xLAM-2-8b-fc-r-gguf'
                repo_filename = '*Q8_0.gguf'
                chat_format = 'chatml-function-calling'
                tool_choice = tools.choice(auto=True)
            case 'mistral':
                repo_id = 'NousResearch/Hermes-2-Pro-Mistral-7B-GGUF'
                repo_filename = '*Q4_K_M.gguf'
                chat_format = 'chatml-function-calling'
                tool_choice = tools.choice(auto=True)
            case 'gemma':
                repo_id = 'lmstudio-community/gemma-3-4b-it-GGUF'
                repo_filename = '*Q8_0.gguf'
                chat_format = 'chatml-function-calling'
                tool_choice = tools.choice(auto=True)
            case _:
                raise AssertionError(f'Unknown model: {model}')

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        messages = [{'role': 'user', 'content': t.prompt}]
        t.add_computed_column(
            response=llama_cpp.create_chat_completion(
                messages,
                repo_id=repo_id,
                repo_filename=repo_filename,
                tools=tools,
                tool_choice=tool_choice,
                chat_format=chat_format,
            )
        )
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
        validate_update_status(t.insert(prompt='What is the weather in San Francisco?'), 1)
        res = t.collect()
        assert res[0]['tool_calls'] == {'weather': ['Cloudy with a chance of meatballs'], 'stock_price': None}, (
            f'Actual row: {res[0]}'
        )
