import pytest

import pixeltable as pxt
from pixeltable.runtime import get_runtime
from tests.functions.tool_utils import run_tool_invocations_test

from ..conftest import SampleFileServer
from ..utils import (
    SAMPLE_IMAGE_FILE_PATH,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


def skip_test_if_no_openllm() -> None:
    # The client needs no credentials, so check that the OpenLLM daemon is reachable
    skip_test_if_not_installed('openai')
    skip_test_if_no_client('openllm')
    import httpx

    try:
        httpx.get(f'{get_runtime().get_client("openllm").base_url}models', timeout=3)
    except httpx.TransportError:
        pytest.skip('OpenLLM daemon is not reachable.')


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestOpenLLM:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openllm import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String | None})
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(chat_output=chat_completions(messages=msgs, model='plus'))
        t.add_computed_column(
            chat_output_2=chat_completions(
                messages=msgs, model='plus', model_kwargs={'max_tokens': 300, 'temperature': 0.7}
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['chat_output_2'][0]['choices'][0]['message']['content'].lower()

    def test_vision(self, uses_db: None, sample_file_server: SampleFileServer) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openllm import chat_completions

        t = pxt.create_table('test_tbl', {'image': pxt.Image | None})
        msgs = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': "What's in this image?"},
                    {'type': 'image_url', 'image_url': t.image},
                ],
            }
        ]
        t.add_computed_column(response=chat_completions(msgs, model='plus'))

        validate_update_status(t.insert(image=sample_file_server.url(SAMPLE_IMAGE_FILE_PATH)), 1)
        result = t.collect()
        assert 'broccoli' in result['response'][0]['choices'][0]['message']['content'].lower()

    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openai import invoke_tools
        from pixeltable.functions.openllm import chat_completions

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String | None}, if_exists='replace')
            t.add_computed_column(
                response=chat_completions(
                    messages=[{'role': 'user', 'content': t.prompt}], model='plus', tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_tool_choice=True)

    def test_responses(self, uses_db: None) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openllm import responses

        t = pxt.create_table('test_tbl', {'input': pxt.String | None})
        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(resp_output=responses(msgs, model='plus'))
        t.add_computed_column(
            resp_output_2=responses(
                msgs,
                model='plus',
                model_kwargs={'instructions': 'You are a helpful assistant.', 'max_output_tokens': 300},
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['resp_output'][0]['output'][0]['content'][0]['text'].lower()
        assert 'paris' in result['resp_output_2'][0]['output'][0]['content'][0]['text'].lower()

    def test_responses_vision(self, uses_db: None, sample_file_server: SampleFileServer) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openllm import responses

        t = pxt.create_table('test_tbl', {'image': pxt.Image | None})
        msgs = [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': "What's in this image?"},
                    {'type': 'input_image', 'image_url': t.image},
                ],
            }
        ]
        t.add_computed_column(response=responses(msgs, model='plus'))

        validate_update_status(t.insert(image=sample_file_server.url(SAMPLE_IMAGE_FILE_PATH)), 1)
        result = t.collect()
        assert 'broccoli' in result['response'][0]['output'][0]['content'][0]['text'].lower()

    def test_responses_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_no_openllm()
        from pixeltable.functions.openai import invoke_tools
        from pixeltable.functions.openllm import responses

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String | None}, if_exists='replace')
            t.add_computed_column(
                response=responses(
                    input=[{'role': 'user', 'content': t.prompt}], model='plus', tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_tool_choice=True)
