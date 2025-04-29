import pytest

import pixeltable as pxt
from tests.conftest import DO_RERUN
from tests.utils import skip_test_if_no_aws_credentials, skip_test_if_not_installed, stock_price


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
        from pixeltable.functions.bedrock import converse, invoke_tools

        tools = pxt.tools(stock_price)
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        t.add_computed_column(
            response=converse(messages, model_id='anthropic.claude-3-haiku-20240307-v1:0', tool_config=tools)
        )
        t.insert(input='What is the stock price of NVDA today?')
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))

        results = t.collect()[0]
        assert results['tool_calls']['stock_price'] == [131.17]
