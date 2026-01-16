import sysconfig

import pytest

import pixeltable as pxt
from pixeltable.config import Config

from ..utils import get_audio_files, runs_linux_with_gpu, skip_test_if_not_installed, validate_update_status


@pytest.mark.skipif(
    sysconfig.get_platform() == 'linux-aarch64', reason='libsndfile.so is missing on CI Linux ARM instances'
)
@pytest.mark.skipif(runs_linux_with_gpu(), reason='crashes on Linux with GPU')
class TestWhisperx:
    def test_transcription(self, uses_store: None) -> None:
        skip_test_if_not_installed('whisperx')
        from pixeltable.functions import whisperx

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisperx', {'audio': pxt.Audio})
        t.add_computed_column(transcription=whisperx.transcribe(t.audio, model='tiny.en'))
        t.add_computed_column(
            transcription2=whisperx.transcribe(
                t.audio, model='tiny.en', compute_type='int8', language='en', task='transcribe', chunk_size=10
            )
        )
        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        results = t.collect()[0]

        assert results['transcription']['language'] == 'en'
        assert 'city upon a hill' in results['transcription']['segments'][1]['text']

        assert results['transcription2']['language'] == 'en'
        assert 'long and deliberate process' in results['transcription2']['segments'][1]['text']
        assert 'city upon a hill' not in results['transcription2']['segments'][1]['text']  # due to shorter chunk size

    def test_diarization(self, uses_store: None) -> None:
        skip_test_if_not_installed('whisperx')
        if Config.get().get_string_value('auth_token', section='hf') is None:
            # Diarization requires a HF access token for the opt-in pyannote models
            pytest.skip('Skipping WhisperX diarization test (no HF_AUTH_TOKEN configured)')
        from pixeltable.functions import whisperx

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisperx', {'audio': pxt.Audio})
        t.add_computed_column(diarization=whisperx.transcribe(t.audio, model='tiny.en', diarize=True))
        t.add_computed_column(
            diarization2=whisperx.transcribe(
                t.audio,
                model='tiny.en',
                diarize=True,
                alignment_model_name='WAV2VEC2_ASR_BASE_960H',
                interpolate_method='linear',
                return_char_alignments=True,
                diarization_model_name='pyannote/speaker-diarization-3.1',
                num_speakers=1,
                min_speakers=1,
                max_speakers=1,
            )
        )
        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        results = t.collect()[0]

        assert results['diarization']['segments'][1]['speaker'] == 'SPEAKER_00'
        assert 'I have been at the task' in results['diarization']['segments'][1]['text']

        assert results['diarization2']['segments'][1]['speaker'] == 'SPEAKER_00'
        assert 'I have been at the task' in results['diarization2']['segments'][1]['text']

    def test_whisperx_errors(self, uses_store: None) -> None:
        skip_test_if_not_installed('whisperx')
        from pixeltable.functions import whisperx

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisperx', {'audio': pxt.Audio}, if_exists='replace')
        t.insert(audio=audio_file)
        for param_name, value in (
            ('alignment_model_name', 'egg'),
            ('interpolate_method', 'linear'),
            ('return_char_alignments', True),
            ('diarization_model_name', 'ham'),
            ('num_speakers', 2),
            ('min_speakers', 1),
            ('max_speakers', 3),
        ):
            with pytest.raises(pxt.Error, match=f'`{param_name}` can only be set if `diarize=True`'):
                t.select(transcription=whisperx.transcribe(t.audio, model='tiny.en', **{param_name: value})).collect()
