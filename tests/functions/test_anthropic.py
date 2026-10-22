import typing

import pytest

import pixeltable as pxt

from ..utils import (
    assert_type_eq,
    rerun,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestAnthropic:
    def test_messages(self, uses_db: None) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions import anthropic

        t = pxt.create_table('test_tbl', {'input': pxt.String | None})
        messages = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=anthropic.messages(messages=messages, model='claude-haiku-4-5', max_tokens=1024))
        t.add_computed_column(
            output2=anthropic.messages(
                messages=messages,
                model='claude-haiku-4-5',
                max_tokens=300,
                # the sampling params (temperature, top_k, top_p) are not part of the messages API
                model_kwargs={
                    'metadata': {'user_id': 'pixeltable'},
                    'stop_sequences': ['STOP'],
                    'system': 'You are an ordinary person walking down the street.',
                    'service_tier': 'auto',
                },
            )
        )
        t.add_computed_column(text=t.output.content[0].text)
        assert_type_eq(t.text.col_type, pxt.String | None)
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['content'][0]['text']) > 0
        assert len(results['output2'][0]['content'][0]['text']) > 0
        assert results['text'][0] == results['output'][0]['content'][0]['text']

        reply = pxt.eval_udf(
            anthropic.messages,
            messages=[{'role': 'user', 'content': 'Reply with one word.'}],
            model='claude-haiku-4-5',
            max_tokens=20,
        )
        assert len(reply['content'][0]['text']) > 0

    @rerun(reruns=6, reruns_delay=8)
    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        from pixeltable.functions import anthropic

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String | None}, if_exists='replace')
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(
                response=anthropic.messages(
                    model='claude-haiku-4-5', messages=messages, max_tokens=1024, tools=tools, tool_choice=tool_choice
                )
            )
            t.add_computed_column(tool_calls=anthropic.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_tool_choice=True)


class TestAnthropicTypedDictAdherence:
    """Detect drift between the TypedDict mirrors and the SDK Pydantic models. Doesn't need credentials."""

    def test_mirror_keys(self) -> None:
        skip_test_if_not_installed('anthropic')
        from anthropic.types import ContentBlock, Message, Usage

        from pixeltable.functions import anthropic

        assert set(anthropic.Message.__annotations__) == set(Message.model_fields)
        assert set(anthropic.Usage.__annotations__) == set(Usage.model_fields)
        # ContentBlock is an Annotated union of the block types
        block_types = typing.get_args(typing.get_args(ContentBlock)[0])
        block_fields = {name for block_type in block_types for name in block_type.model_fields}
        assert set(anthropic.ContentBlock.__annotations__) == block_fields
