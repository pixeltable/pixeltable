import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.tests.utils import skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestFireworks:

    def test_fireworks(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('fireworks')
        TestFireworks.skip_test_if_no_fireworks_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': pxt.StringType()})
        from pixeltable.functions.fireworks import chat_completions
        messages = [{'role': 'user', 'content': t.input}]
        t['output'] = chat_completions(
            messages=messages,
            model='accounts/fireworks/models/llama-v2-7b-chat'
        )
        t['output_2'] = chat_completions(
            messages=messages,
            model='accounts/fireworks/models/llama-v2-7b-chat',
            max_tokens=300,
            top_k=40,
            top_p=0.9,
            temperature=0.7
        )
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['choices'][0]['message']['content']) > 0
        assert len(results['output_2'][0]['choices'][0]['message']['content']) > 0

    # This ensures that the test will be skipped, rather than returning an error, when no API key is
    # available (for example, when a PR runs in CI).
    @staticmethod
    def skip_test_if_no_fireworks_client() -> None:
        try:
            import pixeltable.functions.fireworks
            _ = pixeltable.functions.fireworks.fireworks_client()
        except excs.Error as exc:
            pytest.skip(str(exc))
