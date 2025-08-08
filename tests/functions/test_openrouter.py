import pytest

import pixeltable as pxt

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestOpenRouter:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        # Test basic chat completion
        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=chat_completions(
                messages=msgs,
                model='openai/gpt-3.5-turbo',  # Use a widely available model
            )
        )

        # Test with model kwargs
        t.add_computed_column(
            output2=chat_completions(
                messages=msgs,
                model='anthropic/claude-3-haiku-20240307',
                model_kwargs={'temperature': 0.8, 'max_tokens': 300},
            )
        )

        # Test with provider routing
        t.add_computed_column(
            output3=chat_completions(
                messages=msgs, model='openai/gpt-3.5-turbo', provider={'order': ['OpenAI'], 'allow_fallbacks': True}
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        results = t.collect()

        # Check that all outputs contain responses
        assert len(results['output'][0]['choices'][0]['message']['content']) > 0
        assert 'Paris' in results['output'][0]['choices'][0]['message']['content']
        assert len(results['output2'][0]['choices'][0]['message']['content']) > 0
        assert len(results['output3'][0]['choices'][0]['message']['content']) > 0

    def test_model_listing(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import models

        t = pxt.create_table('models_test', {'fetch': pxt.Bool})
        t.add_computed_column(models_list=models())

        validate_update_status(t.insert(fetch=True), 1)
        results = t.collect()

        models_data = results['models_list'][0]
        assert isinstance(models_data, list)
        assert len(models_data) > 0

        # Check that models have expected fields
        first_model = models_data[0]
        assert 'id' in first_model
        assert 'name' in first_model
        assert 'context_length' in first_model
        assert 'pricing' in first_model

    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions, invoke_tools
        from tests.functions.tool_utils import stock_price

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tool_calls', {'input': pxt.String})

            # Use a model that supports tools
            t.add_computed_column(
                llm_response=chat_completions(
                    messages=[{'role': 'user', 'content': t.input}],
                    model='openai/gpt-4-turbo',
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )
            t.add_computed_column(tool_result=invoke_tools(tools, t.llm_response))
            return t

        # Test 1: Tool invocation using existing stock_price tool
        tools = pxt.tools(stock_price)
        t = make_table(tools, tools.choice(auto=True))
        validate_update_status(t.insert(input='What is the stock price of NVDA?'), 1)

        results = t.collect()
        tool_result = results['tool_result'][0]

        # Check that tool was called
        assert 'stock_price' in tool_result
        price_result = tool_result['stock_price']
        assert price_result == 131.17

    def test_transforms(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions

        t = pxt.create_table('test_transforms', {'text': pxt.String})

        # Test with transforms
        t.add_computed_column(
            summary=chat_completions(
                messages=[{'role': 'user', 'content': f'Summarize: {t.text}'}],
                model='openai/gpt-3.5-turbo',
                transforms=['middle-out'],
            )
            .choices[0]
            .message.content
        )

        # Insert a long text
        long_text = ' '.join([f'Sentence {i}.' for i in range(100)])
        validate_update_status(t.insert(text=long_text), 1)

        results = t.collect()
        summary = results['summary'][0]

        # Check that we got a summary
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(long_text)  # Summary should be shorter
