import pytest

import pixeltable as pxt

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestFireworks:
    def test_fireworks(self, reset_db: None) -> None:
        skip_test_if_not_installed('fireworks')
        skip_test_if_no_client('fireworks')
        from pixeltable.functions.fireworks import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=chat_completions(messages=messages, model='accounts/fireworks/models/mixtral-8x22b-instruct')
        )
        t.add_computed_column(
            output_2=chat_completions(
                messages=messages,
                model='accounts/fireworks/models/mixtral-8x22b-instruct',
                model_kwargs={'max_tokens': 300, 'top_k': 40, 'top_p': 0.9, 'temperature': 0.7},
            )
        )
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['choices'][0]['message']['content']) > 0
        assert len(results['output_2'][0]['choices'][0]['message']['content']) > 0
