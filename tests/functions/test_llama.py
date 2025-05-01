import pytest
import pixeltable as pxt
from pixeltable.functions import llama # Changed import
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status # Removed stock_price
from ..conftest import DO_RERUN # Added
from .tool_utils import run_tool_invocations_test # Added


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN) # Added flaky decorator
class TestLlama:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('llama')
        t = pxt.create_table('test_llama_tbl', {'input': pxt.String})

        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(model=test_model, messages=t.input_msgs))

        t.add_computed_column(
            chat_output_params=chat_completions(
                model=test_model,
                messages=msgs,
                max_tokens=50,
                temperature=0.7,
                top_p=0.8,
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['chat_output_params'][0]['choices'][0]['message']['content'].lower()

        assert 'id' in result['chat_output'][0]
        assert 'model' in result['chat_output'][0]
        assert result['chat_output'][0]['model'].lower().startswith('llama-4-scout-17b-16e-instruct-fp8')
        assert 'usage' in result['chat_output'][0]

    def test_chat_completions_json(self, reset_db: None) -> None:
        """Tests the chat_completions function with JSON response format."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('llama')
        t = pxt.create_table('test_llama_json_tbl', {'input': pxt.String})

        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

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

        t.add_computed_column(
            json_output_raw=chat_completions(model=test_model, messages=msgs, response_format=response_format_arg)
        )

        validate_update_status(t.insert(input='trigger'), 1)
        result = t.collect()

        assert 'choices' in result['json_output_raw'][0]
        message = result['json_output_raw'][0]['choices'][0]['message']
        assert 'content' in message

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
        """Tests the chat_completions function with tool calling using the common test helper."""
        skip_test_if_not_installed('openai')  # For pxt.tools and tool_utils
        skip_test_if_no_client('llama')

        test_model = 'Llama-4-Scout-17B-16E-Instruct-FP8'

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=llama.chat_completions(
                    model=test_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice # Pass ToolChoice object directly
                )
            )
            t.add_computed_column(tool_calls=llama.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_random_question=True)
