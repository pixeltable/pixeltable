from typing import Optional

import numpy as np
import pytest

import pixeltable as pxt
from tests.utils import skip_test_if_not_installed, validate_update_status


class TestOllama:
    @pytest.mark.xdist_group('ollama')
    def test_generate(self, reset_db: None):
        self.__ensure_ollama_availability()
        from pixeltable.functions.ollama import generate

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        # msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=generate(t.input, model='qwen2.5:0.5b'))
        t.add_computed_column(
            output2=generate(
                t.input,
                model='qwen2.5:0.5b',
                options={'temperature': 1.0, 'max_tokens': 300, 'top_p': 0.9, 'top_k': 40},
            )
        )
        validate_update_status(t.insert(input='The average July rainfall in Topeka is '))
        results = t.collect()
        assert len(results['output'][0]['response']) > 0
        assert len(results['output2'][0]['response']) > 0

    @pytest.mark.xdist_group('ollama')
    def test_chat(self, reset_db: None):
        self.__ensure_ollama_availability()
        from pixeltable.functions.ollama import chat

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]

        t.add_computed_column(output=chat(msgs, model='qwen2.5:0.5b'))
        t.add_computed_column(
            output2=chat(
                msgs, model='qwen2.5:0.5b', options={'temperature': 1.0, 'max_tokens': 300, 'top_p': 0.9, 'top_k': 40}
            )
        )
        validate_update_status(t.insert(input='What are the spiciest varieties of peppers?'))
        results = t.collect()
        assert len(results['output'][0]['message']['content']) > 0
        assert len(results['output2'][0]['message']['content']) > 0

    @pytest.mark.xdist_group('ollama')
    def test_embed(self, reset_db: None):
        self.__ensure_ollama_availability()
        from pixeltable.functions.ollama import embed

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t.add_computed_column(output=embed(t.input, model='qwen2.5:0.5b'))
        validate_update_status(t.insert(input='I am a purple cloud.'))
        results = t.collect()
        assert isinstance(results['output'][0], np.ndarray)
        assert len(results['output'][0]) == 896

    def __ensure_ollama_availability(self):
        skip_test_if_not_installed('ollama')
        if self.__ollama_available is None:
            import ollama

            try:
                ollama.pull('qwen2.5:0.5b')
                ollama.generate(model='qwen2.5:0.5b', prompt='Are you properly configured?')
                self.__ollama_available = True
            except Exception as exc:
                self.__ollama_available = False
                self.__ollama_exception = exc
        if not self.__ollama_available:
            pytest.skip(f'ollama not available: {self.__ollama_exception}')

    __ollama_available: Optional[bool] = None
    __ollama_exception: Optional[Exception] = None
