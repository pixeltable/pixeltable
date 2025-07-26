import pytest

import pixeltable as pxt
from pixeltable.tests.utils import skip_test_if_no_client, skip_test_if_not_installed


class TestOpenRouter:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions

        t = pxt.create_table('test_openrouter', {'input': pxt.String})
        t.insert({'input': 'What is the capital of France?'})

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': t.input}
        ]
        t.add_computed_column(output=chat_completions(messages, model='openai/gpt-4o-mini'))
        result = t.select(t.output).collect()
        assert len(result) == 1
        assert 'choices' in result[0]['output']
        assert len(result[0]['output']['choices']) > 0
        assert 'message' in result[0]['output']['choices'][0]
        assert 'content' in result[0]['output']['choices'][0]['message']
        # The response should mention Paris
        assert 'Paris' in result[0]['output']['choices'][0]['message']['content']

    def test_chat_completions_with_attribution(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions

        t = pxt.create_table('test_openrouter_attr', {'input': pxt.String})
        t.insert({'input': 'Hello world'})

        messages = [
            {'role': 'user', 'content': t.input}
        ]
        t.add_computed_column(
            output=chat_completions(
                messages, 
                model='openai/gpt-4o-mini',
                site_url='https://pixeltable.com',
                site_name='Pixeltable Test'
            )
        )
        result = t.select(t.output).collect()
        assert len(result) == 1
        assert 'choices' in result[0]['output']
        assert len(result[0]['output']['choices']) > 0

    def test_chat_completions_with_tools(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions, invoke_tools

        t = pxt.create_table('test_openrouter_tools', {'input': pxt.String})
        t.insert({'input': 'What is 15 + 27?'})

        # Define a simple calculator tool
        @pxt.udf
        def calculator(operation: str, a: float, b: float) -> dict:
            if operation == 'add':
                return {'result': a + b}
            elif operation == 'subtract':
                return {'result': a - b}
            elif operation == 'multiply':
                return {'result': a * b}
            elif operation == 'divide':
                return {'result': a / b if b != 0 else 'Error: Division by zero'}
            else:
                return {'result': 'Error: Unknown operation'}

        tools = pxt.func.Tools([calculator])

        messages = [
            {'role': 'user', 'content': t.input}
        ]
        
        t.add_computed_column(
            response=chat_completions(
                messages, 
                model='openai/gpt-4o-mini',
                tools=tools.tool_specs,
                tool_choice={'auto': True, 'required': False, 'tool': None, 'parallel_tool_calls': True}
            )
        )
        
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
        
        result = t.select(t.response, t.tool_calls).collect()
        assert len(result) == 1
        assert 'choices' in result[0]['response']

    def test_different_models(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openrouter import chat_completions

        t = pxt.create_table('test_openrouter_models', {'input': pxt.String})
        t.insert({'input': 'Say hello'})

        messages = [{'role': 'user', 'content': t.input}]
        
        # Test with different model providers available through OpenRouter
        models_to_test = [
            'openai/gpt-4o-mini',
            'anthropic/claude-3-haiku',
            'meta-llama/llama-3.1-8b-instruct:free'
        ]

        for i, model in enumerate(models_to_test):
            column_name = f'output_{i}'
            t.add_computed_column(**{column_name: chat_completions(messages, model=model)})

        result = t.select().collect()
        assert len(result) == 1
        
        # Check that all model responses have the expected structure
        for i in range(len(models_to_test)):
            column_name = f'output_{i}'
            assert column_name in result[0]
            assert 'choices' in result[0][column_name]
            assert len(result[0][column_name]['choices']) > 0