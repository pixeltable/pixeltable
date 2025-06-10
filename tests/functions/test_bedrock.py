import pytest

import pixeltable as pxt

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_aws_credentials, skip_test_if_not_installed
from .tool_utils import run_tool_invocations_test


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestBedrock:
    def test_converse(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        system = [
            {
                'text': 'You are an app that creates playlists for a radio station that plays rock and pop music. '
                'Only return song names and the artist.'
            }
        ]

        t.add_computed_column(output=converse(messages, model_id='anthropic.claude-3-haiku-20240307-v1:0'))
        t.add_computed_column(
            output2=converse(
                messages,
                model_id='anthropic.claude-3-haiku-20240307-v1:0',
                system=system,
                inference_config={'temperature': 0.6},
                additional_model_request_fields={'top_k': 40},
            )
        )
        t.add_computed_column(response=t.output.output.message.content[0].text)
        t.add_computed_column(response2=t.output.output.message.content[0].text)

        t.insert(input='What were the 10 top charting pop singles in 2010?')
        results = t.collect()[0]
        assert 'Katy Perry' in results['response']
        assert 'Katy Perry' in results['response2']

    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions import bedrock

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String})
            messages = [{'role': 'user', 'content': [{'text': t.prompt}]}]
            t.add_computed_column(
                response=bedrock.converse(
                    messages, model_id='anthropic.claude-3-haiku-20240307-v1:0', tool_config=tools
                )
            )
            t.add_computed_column(tool_calls=bedrock.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_multiple_tool_use=False)
