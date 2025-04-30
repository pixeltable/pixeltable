import pytest

import pixeltable as pxt

# from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
# @pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN) # Add back if needed
class TestLlama:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('llama')
        t = pxt.create_table('test_llama_tbl', {'input': pxt.String})
        from pixeltable.functions.llama import chat_completions

        # Use a small, fast model available via the API for testing
        # Note: Model availability might change, adjust if needed.
        # Llama-3.1-8B-Instruct seems like a reasonable choice if available.
        # Using Llama-3.3-8B-Instruct as per the example in the docs.
        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(model=test_model, messages=t.input_msgs))

        # Test with some parameters
        t.add_computed_column(
            chat_output_params=chat_completions(
                model=test_model,
                messages=msgs,
                max_tokens=50,  # Keep small for testing
                temperature=0.7,
                top_p=0.8,
            )
        )

        # TODO: Add tests for other features like JSON mode, tool use if applicable and easy to test.

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['chat_output_params'][0]['choices'][0]['message']['content'].lower()

        # Check presence of expected keys
        assert 'id' in result['chat_output'][0]
        assert 'model' in result['chat_output'][0]
        # Check model name, allow for variations
        assert result['chat_output'][0]['model'].lower().startswith('llama-4-scout-17b-16e-instruct-fp8')
        assert 'usage' in result['chat_output'][0]

    def test_chat_completions_json(self, reset_db: None) -> None:
        """Tests the chat_completions function with JSON response format."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('llama')
        import json

        t = pxt.create_table('test_llama_json_tbl', {'input': pxt.String})
        from pixeltable.functions.llama import chat_completions

        # Use a model confirmed to work with JSON mode
        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

        # Define the prompt and expected schema
        json_prompt = 'Extract the name and day from: Alice went to the park on Friday.'
        event_schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'day_of_week': {'type': 'string'}},
            'required': ['name', 'day_of_week'],
        }
        response_format_arg = {'type': 'json_schema', 'json_schema': {'schema': event_schema}}

        msgs = [
            {'role': 'system', 'content': 'Extract information into the specified JSON format.'},
            {'role': 'user', 'content': json_prompt},
        ]

        # Note: messages are static here, not dependent on t.input for this specific test
        t.add_computed_column(
            json_output_raw=chat_completions(model=test_model, messages=msgs, response_format=response_format_arg)
        )

        # Insert a dummy row to trigger computation (input value doesn't matter here)
        validate_update_status(t.insert(input='trigger'), 1)
        result = t.collect()

        # Check the raw output structure
        assert 'choices' in result['json_output_raw'][0]
        message = result['json_output_raw'][0]['choices'][0]['message']
        assert 'content' in message

        # Validate the content is a valid JSON string matching the expected structure
        try:
            parsed_content = json.loads(message['content'])
            assert isinstance(parsed_content, dict)
            assert 'name' in parsed_content
            assert parsed_content['name'].lower() == 'alice'
            assert 'day_of_week' in parsed_content
            assert parsed_content['day_of_week'].lower() == 'friday'
        except json.JSONDecodeError:
            pytest.fail(f'Output content is not valid JSON: {message["content"]}')
        except AssertionError as e:
            pytest.fail(f'JSON content validation failed: {e}. Content was: {message["content"]}')

    def test_tool_invocations(self, reset_db: None) -> None:
        """Tests the chat_completions function with tool calling."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('llama')
        from pixeltable.functions.llama import chat_completions, invoke_tools
        from ..utils import stock_price  # Import dummy tool from test utils

        # Register the tool
        tools_pxt = pxt.tools(stock_price)

        # Use a model confirmed to work with standard tool calls
        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

        t = pxt.create_table('test_llama_tool_tbl', {'prompt': pxt.String})
        messages = [{'role': 'user', 'content': t.prompt}]

        # 1. Get LLM request for tool call
        t.add_computed_column(llm_tool_request=chat_completions(model=test_model, messages=messages, tools=tools_pxt))

        # 2. Invoke the tool based on the request
        t.add_computed_column(tool_output=invoke_tools(tools=tools_pxt, response=t.llm_tool_request))

        # Insert prompt designed to trigger the tool
        validate_update_status(t.insert(prompt='What is the stock price of NVDA?'), 1)
        result = t.collect()

        # Check LLM request
        request_data = result['llm_tool_request'][0]
        assert 'choices' in request_data
        message = request_data['choices'][0]['message']
        assert 'tool_calls' in message
        assert message['tool_calls'] is not None
        assert len(message['tool_calls']) == 1
        tool_call = message['tool_calls'][0]
        assert tool_call['type'] == 'function'
        assert tool_call['function']['name'] == 'stock_price'
        assert 'NVDA' in tool_call['function']['arguments']  # Basic check for argument presence

        # Check tool execution output
        tool_output_data = result['tool_output'][0]
        assert isinstance(tool_output_data, dict)
        assert 'stock_price' in tool_output_data
        # Check if the tool returned the expected value (list because invoke_tools aggregates results)
        assert tool_output_data['stock_price'] == [131.17]
