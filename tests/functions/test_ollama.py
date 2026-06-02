import numpy as np
import pytest

import pixeltable as pxt
from tests.utils import skip_test_if_not_installed, validate_update_status


_ollama_available: bool | None = None
_ollama_exception: Exception | None = None

def _ensure_ollama_availability() -> None:
    global _ollama_available, _ollama_exception
    skip_test_if_not_installed('ollama')
    if _ollama_available is None:
        import ollama

        try:
            ollama.pull('qwen3:0.6b')
            ollama.pull('qwen3-embedding:0.6b')
            ollama.generate(model='qwen3:0.6b', prompt='Are you properly configured?')
            _ollama_available = True
        except Exception as exc:
            _ollama_available = False
            _ollama_exception = exc
    if not _ollama_available:
        pytest.skip(f'ollama not available: {_ollama_exception}')

@pytest.mark.expensive
@pytest.mark.xdist_group('ollama')
class TestOllama:
    def test_generate(self, uses_db: None) -> None:
        _ensure_ollama_availability()
        from pixeltable.functions.ollama import generate

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        # msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=generate(t.input, model='qwen3:0.6b'))
        t.add_computed_column(
            output2=generate(
                t.input,
                model='qwen3:0.6b',
                options={'temperature': 1.0, 'max_tokens': 300, 'top_p': 0.9, 'top_k': 40},
            )
        )
        validate_update_status(t.insert(input='The average July rainfall in Topeka is '))
        results = t.collect()
        assert len(results['output'][0]['response']) > 0
        assert len(results['output2'][0]['response']) > 0

    def test_chat(self, uses_db: None) -> None:
        _ensure_ollama_availability()
        from pixeltable.functions.ollama import chat

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]

        t.add_computed_column(output=chat(msgs, model='qwen3:0.6b'))
        t.add_computed_column(
            output2=chat(
                msgs, model='qwen3:0.6b', options={'temperature': 1.0, 'max_tokens': 300, 'top_p': 0.9, 'top_k': 40}
            )
        )
        validate_update_status(t.insert(input='What are the spiciest varieties of peppers?'))
        results = t.collect()
        assert len(results['output'][0]['message']['content']) > 0
        assert len(results['output2'][0]['message']['content']) > 0

    def test_embed(self, uses_db: None) -> None:
        _ensure_ollama_availability()
        from pixeltable.functions.ollama import embed

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t.add_computed_column(output=embed(t.input, model='qwen3-embedding:0.6b'))
        validate_update_status(t.insert(input='I am a purple cloud.'))
        results = t.collect()
        assert isinstance(results['output'][0], np.ndarray)
        assert len(results['output'][0]) == 1024
