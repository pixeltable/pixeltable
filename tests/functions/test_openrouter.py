import pytest

import pixeltable as pxt
from tests.functions.tool_utils import run_tool_invocations_test

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
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
                model='anthropic/claude-sonnet-4.5',
                model_kwargs={'temperature': 0.8, 'max_tokens': 300},
                provider={'order': ['Anthropic'], 'allow_fallbacks': True},
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        results = t.collect()

        # Check that all outputs contain responses
        assert 'Paris' in results['output'][0]['choices'][0]['message']['content']
        assert 'Paris' in results['output2'][0]['choices'][0]['message']['content']

    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('openrouter')
        from pixeltable.functions.openai import invoke_tools
        from pixeltable.functions.openrouter import chat_completions

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')

            # Use a model that supports tools
            t.add_computed_column(
                response=chat_completions(
                    messages=[{'role': 'user', 'content': t.prompt}],
                    model='openai/gpt-4-turbo',
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )
            t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table)

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
