import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from ..utils import skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8)
class TestFireworks:
    def test_fireworks(self, reset_db) -> None:
        skip_test_if_not_installed('fireworks')
        TestFireworks.skip_test_if_no_fireworks_client()
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        from pixeltable.functions.fireworks import chat_completions

        messages = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=chat_completions(messages=messages, model='accounts/fireworks/models/mixtral-8x22b-instruct')
        )
        t.add_computed_column(
            output_2=chat_completions(
                messages=messages,
                model='accounts/fireworks/models/mixtral-8x22b-instruct',
                max_tokens=300,
                top_k=40,
                top_p=0.9,
                temperature=0.7,
            )
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

            _ = pixeltable.functions.fireworks._fireworks_client()
        except excs.Error as exc:
            pytest.skip(str(exc))
